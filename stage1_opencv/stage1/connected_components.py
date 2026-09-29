"""连通域分析 + 外轮廓提取：把"一团白像素"变成一个可管理的对象。

对应主仓库：
- 监督 eval 的输出链：mask -> 连通域统计 -> 检出组件 bbox + 面积（summary.csv）；
- 无监督 eval：closed_mask -> findContours -> drawContours 红框高亮；
- 面积过滤 = 业务上的"最小缺陷面积门限"（eval --min_area 默认 100px）。

运行：python -m stage1.connected_components
"""

import cv2
import numpy as np


def label_components(binary: np.ndarray):
    """cv2.connectedComponentsWithStats：一次拿到标签图 + 每域统计。

    返回 (num, labels, stats, centroids)：
      num       连通域个数（含背景 0 号）
      labels    与输入同尺寸的 int32 图，每个像素属于哪个域
      stats     (num, 5)：[left, top, width, height, area]（注意前四项是 x,y,w,h）
      centroids (num, 2)：每个域的质心 (cx, cy)
    """
    return cv2.connectedComponentsWithStats(binary, connectivity=8)


def components_report(binary: np.ndarray, min_area: int = 0):
    """把 stats 整理成字典列表，并按 min_area 过滤——质检流水线的标准动作。"""
    num, _, stats, centroids = label_components(binary)
    comps = []
    for i in range(1, num):                          # 跳过 0 号背景
        left, top, width, height, area = stats[i]
        if area < min_area:
            continue
        comps.append({
            "id": i, "left": int(left), "top": int(top),
            "w": int(width), "h": int(height), "area": int(area),
            "cx": float(centroids[i][0]), "cy": float(centroids[i][1]),
        })
    comps.sort(key=lambda c: -c["area"])             # 大缺陷排前面
    return comps


def filter_by_area(binary: np.ndarray, min_area: int) -> np.ndarray:
    """只保留面积 >= min_area 的连通域，输出过滤后的二值图。"""
    num, labels, stats, _ = label_components(binary)
    keep = np.zeros_like(binary)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == i] = 255
    return keep


def external_contours(binary: np.ndarray):
    """外轮廓版统计：findContours + boundingRect + contourArea。

    与 connectedComponents 的分工（仓库两处都在用）：
      - connectedComponents：要"每一团"的编号/面积/质心 -> 面积过滤、统计表；
      - findContours：要"外边缘折线" -> 画红框高亮、算周长。
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        results.append({
            "bbox": (int(x), int(y), int(w), int(h)),
            "area": float(cv2.contourArea(contour)),   # 折线面积，略小于像素面积
            "perimeter": float(cv2.arcLength(contour, True)),
        })
    results.sort(key=lambda c: -c["area"])
    return results


def draw_boxes(image_bgr: np.ndarray, comps, color=(0, 0, 255), thickness=2):
    """在原图上画检出框（对应 eval 的 __merged 红框三联图）。"""
    out = image_bgr.copy()
    for c in comps:
        x, y = c.get("left"), c.get("top")
        w, h = c.get("w"), c.get("h")
        if w is None:                                 # contour 版输入
            x, y, w, h = c["bbox"]
        cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
    return out


# ---------------------------------------------------------------------------
# 演示入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 64)
    print("1) 连通域统计：一团白像素 = 一个对象")
    print("-" * 64)
    binary = np.zeros((100, 120), np.uint8)
    cv2.rectangle(binary, (10, 10), (49, 49), 255, -1)    # 40x40 方块
    cv2.circle(binary, (90, 30), 15, 255, -1)             # r=15 圆
    cv2.rectangle(binary, (30, 70), (89, 89), 255, -1)    # 60x20 长条
    for y, x in [(55, 55), (60, 100), (5, 110)]:          # 三个孤立噪点
        binary[y, x] = 255

    print(f"{'id':>3} {'left':>5} {'top':>5} {'w':>4} {'h':>4} {'area':>6}  质心")
    for c in components_report(binary):
        print(f"{c['id']:>3} {c['left']:>5} {c['top']:>5} {c['w']:>4} "
              f"{c['h']:>4} {c['area']:>6}  ({c['cx']:.1f}, {c['cy']:.1f})")
    print("（前三行是大块；孤立噪点 area=1 也各有自己的域）")

    print("=" * 64)
    print("2) 面积过滤：业务上的最小缺陷尺寸门限")
    print("-" * 64)
    filtered = filter_by_area(binary, min_area=100)
    n_before = len(components_report(binary))
    n_after = len(components_report(filtered))
    print(f"min_area=100 过滤前 {n_before} 个域 -> 过滤后 {n_after} 个"
          f"（3 个单像素噪点被剔除，对应 eval --min_area 的语义）")

    print("=" * 64)
    print("3) 外轮廓版：bbox / 折线面积 / 周长")
    print("-" * 64)
    for c in external_contours(filtered):
        print(f"bbox={c['bbox']}  像素面积≈{c['area']:.0f}  周长={c['perimeter']:.1f}")
    print("（折线面积略小于像素面积：4x4 方块折线面积 9 vs 像素 16——"
          "量尺寸时想清楚要哪个口径，对应仓库'标注口径'问题的几何版）")
