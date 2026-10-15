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
    """把"一堆白格子"整理成"带编号的对象清单"——二值图 -> 编号图（连通域分析）。

    核心就一件事：把黑白图翻译成"编号图"——每个格子里的数字 = 它属于第几团。
    人一眼看得出下图有 3 团，程序不知道；这个函数负责数出来、编上号：

        原图（#=白）      labels（每个数 = 该格属于几号团）
        ........          0 0 0 0 0 0 0 0   <- 0 号 = 背景（所有黑格子）
        .##..##.          0 1 1 0 0 2 2 0   <- 左上 2x2 = 1 号
        .##..##.          0 1 1 0 0 2 2 0
        ........          0 0 0 0 0 0 0 0
        #.......          3 0 0 0 0 0 0 0   <- 左下孤点 = 3 号

    "挨着"就算一团：上下左右 + 斜角都算（connectivity=8；实测斜角串 8 连通算 1 团、
    4 连通算 3 团——写死 8 是为了统一口径，否则两人数出的缺陷个数可能对不上）。
    发号顺序 = 从上到下、从左到右扫描，谁先被碰到谁先发号，没有别的含义。

    返回 (num, labels, stats, centroids) = 围绕编号图的四个答案：
      num       一共几团——含背景 0 号，真目标数 = num - 1（所以循环从 1 开始）
      labels    上面的编号图（int32）；也是反查表：labels == i 就是"只含 i 号团"
                的图——filter_by_area 靠它把某一团整团保留或扔掉
      stats     (num, 5)：每团 [left, top, w, h, area]。left/top 是包围盒左上角
                坐标 (x, y)（不是左/上/右/下）；area 是白格子个数；例：1 号团 =
                [1, 1, 2, 2, 4]。0 号（背景）行 bbox 覆盖整幅图、area 含白块内部
                的洞（洞不算独立域），统计时从不看它
      centroids (num, 2)：每团中心 (cx, cy) = 白格坐标的平均；例 1 号 = (1.5, 1.5)
                ——会有半个像素的小数

    业务位置：质检报告要写"检出 3 个缺陷、每个在哪、多大"，mask 只有像素，这份
    清单就是它的活（-> summary.csv）。实现 = cv2.connectedComponentsWithStats，
    唯一动的手脚是把 connectivity 写死为 8。
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
    """外轮廓版统计：换个口径数缺陷——connectedComponents 是"数格子"，这里是
    "描边"：把每一团白像素的外边界描成一条闭合折线（当成多边形），量出 bbox、
    折线面积、周长。

    返回 [{"bbox": (x,y,w,h), "area": 折线面积, "perimeter": 周长}, ...]，按面积
    降序；没有 id/质心，也没有 min_area 过滤——它只负责"量"，不负责"筛"。

    ★ 两种口径实测对照（同一张图，像素口径见 components_report）：
                        像素口径    折线口径（这里）
        40×40 方块        1600      1521 = 39×39（周长 156 = 4×39）
        60×20 长条        1200      1121 = 59×19
        圆 r=15           709≈πr²   666（周长 98.9 > 2πr≈94.2，锯齿）
        单像素            1         0（周长也是 0）
      为什么小一圈：折线沿"像素中心"走——40 像素宽的两端中心只隔 39 格。最狠的
      是单像素：折线只有 1 个点、围不出面积，contourArea = 0——拿它做 min_area
      过滤会出事，两种口径别混用。

    ★ 逐行要点：
      findContours(binary, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
        RETR_EXTERNAL       只要最外圈轮廓、不管白块内部的洞——函数名的由来
        CHAIN_APPROX_SIMPLE 折线压缩、直边只留拐点（实测：40×40 方块 -> 4 个点，
                            圆 -> 52 个点）；返回的 contours 顺序不保证（所以末尾
                            要 sort）；一条 contour 是 (N,1,2) 的 (x,y) 点列
      boundingRect(contour)   套"最小正矩形"（轴对齐），返回 (x,y,w,h)——和 stats
                              同口径，可直接喂 draw_boxes
      contourArea / arcLength(c, True)   多边形面积 / 闭合折线总长（True=首尾连上）

    分工（仓库两处都在用）：要像素口径的面积、统计表、summary.csv 用
    connectedComponents（见 components_report）；要画红框高亮、算周长、轮廓几何
    用这里（无监督 eval 的红框三联图）。口径警示：同一张图两人量出的缺陷面积可能
    差一成（1600 vs 1521）——不是 bug 是口径，定验收标准前先对齐"这 1 像素算不算"。
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
