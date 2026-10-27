"""课时 2：监控页按需推理渲染的三层 LRU 缓存 —— web.py::Renderer 复刻。

监控网页要让工程师"点开任意一轮权重、看它在任意 NG 图上的热力图"，但整图推理
很贵。真工程 Renderer 的缓存设计（三层 LRU + 只重画不重推 + 去抖）：

  1. 模型层（≤2 份，key=latest/epN，mtime 失效）：权重文件被新一轮覆盖时 mtime
     变化 → 自动重载，且**级联清空**该权重名下的全部热力图缓存（旧权重算的图
     全部作废）；LRU 只留 2 份（显存友好）；
  2. 热力图层（≤3 张，key=(权重, 点位, 图名)）：整图推理结果缓存——改阈值查看
     **只重画不重推**（阈值是渲染参数不是推理参数）；
  3. 底图层（≤6 张）：标注底图（灰度+检测区+标注框）JPEG 缓存；
  4. 悬停查值（/api/heatval）：鼠标处的**全分辨率**精确置信度——热力图被下采样
     显示，肉眼读数与真值有差；120ms 去抖防高频查询。

对应主仓库：src/train/web.py::Renderer（_model / _heat / _vis_base / render /
heat_val 段）、web_page.py（heatVal 去抖段）。

运行：python -m stage11.render_cache
"""
import time
from collections import OrderedDict

import numpy as np


def fake_predict(model_id: str, image_id: str) -> np.ndarray:
    """替身整图推理：生成一张与 (权重, 图) 绑定的"热力图"（真工程 1~2s/张）。"""
    rng = np.random.default_rng(hash((model_id, image_id)) % (2**31))
    heat = rng.random((64, 64))
    heat[20:30, 20:30] += 0.8                          # 假装有个缺陷
    return np.clip(heat, 0, 1)


class Renderer:
    """三层 LRU 渲染缓存：模型(mtime 失效) / 热力图(阈值只重画) / 底图。"""

    MODEL_N = 2
    HEAT_N = 3
    VBASE_N = 6

    def __init__(self, predict_fn=fake_predict):
        self.predict_fn = predict_fn
        self._models: OrderedDict = OrderedDict()       # key → (mtime, id)
        self._heats: OrderedDict = OrderedDict()        # (key, img) → heat
        self._vbase: OrderedDict = OrderedDict()        # img → 底图
        self.counters = {"predict": 0, "render": 0, "model_load": 0}

    def model_mtime(self, key: str, mtimes: dict) -> float:
        return mtimes.get(key, 0.0)

    def _model(self, key: str, mtimes: dict):
        """模型层：mtime 失效 + 级联清空该权重的热力图 + LRU≤2。"""
        mtime = self.model_mtime(key, mtimes)
        cached = self._models.get(key)
        if cached is None or cached[0] != mtime:
            self.counters["model_load"] += 1
            for hk in [k for k in self._heats if k[0] == key]:   # 级联失效
                del self._heats[hk]
            while len(self._models) >= self.MODEL_N:
                self._models.popitem(last=False)
            self._models[key] = (mtime, key)
        self._models.move_to_end(key)
        return self._models[key][1]

    def _heat(self, key: str, image_id: str, mtimes: dict):
        """热力图层：LRU≤3；命中则零推理。"""
        self._model(key, mtimes)                        # 先保证模型有效
        ck = (key, image_id)
        if ck not in self._heats:
            self.counters["predict"] += 1
            self._heats[ck] = self.predict_fn(key, image_id)
            while len(self._heats) > self.HEAT_N:
                self._heats.popitem(last=False)
        self._heats.move_to_end(ck)
        return self._heats[ck]

    def render(self, key: str, image_id: str, threshold: float, mtimes: dict):
        """渲染：阈值只影响上色（重画），推理结果来自热力图缓存。"""
        heat = self._heat(key, image_id, mtimes)
        self.counters["render"] += 1
        binary = heat > threshold
        overlay = (heat * 255).astype(np.uint8)
        overlay[binary] = 255
        return overlay

    def vbase(self, image_id: str):
        """底图层：LRU≤6（真工程存 JPEG 编码后的标注底图）。"""
        if image_id not in self._vbase:
            self._vbase[image_id] = np.full((64, 64), 128, dtype=np.uint8)
            while len(self._vbase) > self.VBASE_N:
                self._vbase.popitem(last=False)
        self._vbase.move_to_end(image_id)
        return self._vbase[image_id]

    def heat_val(self, key: str, image_id: str, y: int, x: int, mtimes: dict) -> float:
        """悬停查值：全分辨率精确置信度（下采样显示图读不准）。"""
        return float(self._heat(key, image_id, mtimes)[y, x])

    def reset_cache(self):
        self._models.clear()
        self._heats.clear()
        self._vbase.clear()


class Debouncer:
    """120ms 去抖：悬停高频触发只在停顿时真正执行（web_page.heatVal 同款）。"""

    def __init__(self, wait_ms: int = 120):
        self.wait = wait_ms / 1000.0
        self._last_run = 0.0

    def should_run(self, now: float) -> bool:
        if now - self._last_run >= self.wait:
            self._last_run = now
            return True
        return False


if __name__ == "__main__":
    r = Renderer()
    mtimes = {"ep1": 1.0, "ep2": 1.0, "ep3": 1.0}

    # 1. 改阈值只重画不重推
    r.render("ep1", "ng_001", 0.5, mtimes)
    r.render("ep1", "ng_001", 0.3, mtimes)
    r.render("ep1", "ng_001", 0.7, mtimes)
    print(f"同一 (权重,图) 渲染 3 次不同阈值: 推理 {r.counters['predict']} 次, "
          f"重画 {r.counters['render']} 次 (只重画不重推)")

    # 2. 热力图 LRU ≤ 3
    for img in ("ng_001", "ng_002", "ng_003", "ng_004"):
        r.render("ep1", img, 0.5, mtimes)
    print(f"查看 4 张图: 热力图缓存 {len(r._heats)} 张 (≤3, 最旧的 ng_001 被挤出)")

    # 3. 模型 mtime 失效级联
    r.render("ep2", "ng_002", 0.5, mtimes)
    mtimes["ep2"] = 2.0                                 # 新一轮覆盖了权重
    r.render("ep2", "ng_002", 0.5, mtimes)
    print(f"权重被覆盖 (mtime 变化): 模型重载 {r.counters['model_load']} 次, "
          f"该权重旧热力图全部作废 → 总推理 {r.counters['predict']} 次")

    # 4. 模型 LRU ≤ 2
    for k in ("ep1", "ep2", "ep3"):
        r.render(k, "ng_002", 0.5, mtimes)
    print(f"轮换 3 个权重: 模型缓存 {len(r._models)} 份 (≤2, 显存友好)")

    # 5. 悬停查值 + 去抖
    val = r.heat_val("ep3", "ng_002", 25, 25, mtimes)
    d = Debouncer(120)
    now = 1000.0
    fires = sum(d.should_run(now + i * 0.02) for i in range(10))   # 200ms 内 10 次
    print(f"悬停查值: 缺陷中心置信度 {val:.3f} | 去抖 200ms 内 10 次触发只执行 {fires} 次")
