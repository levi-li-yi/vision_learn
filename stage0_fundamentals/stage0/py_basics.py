"""Python 语言要点：函数 / 类 / 装饰器 / 类型标注 / dataclass 配置风格。

为什么学这些：vision-ai-training 的全部代码风格 = 类型标注函数 + dataclass 配置
（confs/training_config.py 的 TrainingArgs 就是 @dataclass），CLI 参数从 dataclass
自动生成。把本文件的每个概念吃透，回去读仓库的 confs/ 和 base_utils/parse_utils.py
就不会有语言层面的障碍。

运行方式：
    python -m stage0.py_basics
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# 1. 函数与类型标注
# ---------------------------------------------------------------------------

def parse_csv(raw: Optional[str]) -> List[str]:
    """把逗号分隔的字符串解析成列表——仓库 train_entry.py 里 _parse_csv 的简化版。

    类型标注读法：参数 raw 是 str 或 None；返回 List[str]。
    这是仓库 CLI 白名单参数（--defect_whitelist "砂眼, 崩边"）的解析链路。
    """
    if raw is None:
        return []          # 不传 = 不过滤
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_csv_int(raw: Optional[str]) -> List[int]:
    """逗号分隔整数解析（对应仓库 --region_whitelist "1,2"）。"""
    return [int(item) for item in parse_csv(raw)]


# ---------------------------------------------------------------------------
# 2. 类：封装"图"这个概念（迷你版，为阶段 1 的真实图像处理热身）
# ---------------------------------------------------------------------------

class GrayImage:
    """一张灰度图 = 二维 uint8 数组。

    属性用下划线命名 + property 暴露，是仓库代码的常见写法。
    """

    def __init__(self, pixels):
        pixels = np.asarray(pixels)
        if pixels.ndim != 2:
            raise ValueError(f"灰度图必须是二维数组，收到 ndim={pixels.ndim}")
        if pixels.min() < 0 or pixels.max() > 255:
            raise ValueError("像素值必须落在 [0, 255]")

        self._pixels = pixels.astype(np.uint8)

    @property
    def shape(self):
        """(高, 宽)——注意顺序是 行在前。"""
        return self._pixels.shape

    @property
    def mean_brightness(self):
        """平均亮度：NumPy 归约的第一次亮相。"""
        return float(self._pixels.mean())

    def binarize(self, threshold: int):
        """二值化：大于阈值为 1，否则为 0——工业质检里"判好判坏"的最底层操作。"""
        return (self._pixels > threshold).astype(np.uint8)

    def __repr__(self):
        return f"GrayImage(shape={self.shape}, mean={self.mean_brightness:.1f})"


# ---------------------------------------------------------------------------
# 3. 装饰器：不改动原函数就给它加能力
# ---------------------------------------------------------------------------

def count_calls(func: Callable):
    """调用计数装饰器：训练日志里"这是第几个 step"就是类似机制。"""
    calls = {"n": 0}       # 闭包捕获（dict 可变，函数内才能修改）

    def wrapper(*args, **kwargs):
        calls["n"] += 1
        result = func(*args, **kwargs)
        print(f"    [count_calls] {func.__name__} 第 {calls['n']} 次调用完成")
        return result

    return wrapper


def memoize(func: Callable):
    """记忆化装饰器：缓存已算过的结果。

    应用：无监督线的 teacher 输出对同一张图永远相同（权重冻结），
    理论上就是"可以缓存"的——装饰器是表达这种语义的语言工具。
    """
    cache = {}

    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]

    wrapper.cache = cache     # 暴露缓存便于测试
    return wrapper


@memoize
def slow_square(x: int) -> int:
    """假装很慢的平方函数（每次 sleep 模拟计算成本）。"""
    import time
    time.sleep(0.01)
    return x * x


# ---------------------------------------------------------------------------
# 4. dataclass：仓库配置系统的核心（对照 confs/training_config.py）
# ---------------------------------------------------------------------------

@dataclass
class TrainingArgs:
    """迷你版训练配置——仓库同名类的子集。

    dataclass 让你只声明字段，__init__/__repr__/__eq__ 自动生成。
    仓库的 CustomArgumentParser 就是把这样的类变成 CLI 参数。
    """
    epochs: int = 25
    lr: float = 4e-4
    train_mode: str = "fresh"
    task_whitelist: str = "all"
    defect_whitelist: str = "砂眼, 崩边, 缺肉"   # 默认值与仓库一致
    extras: List[str] = field(default_factory=list)  # 可变默认值必须用 field

    def summary(self) -> str:
        return (f"mode={self.train_mode}, epochs={self.epochs}, "
                f"lr={self.lr:.0e}, 白名单={self.defect_whitelist}")


def apply_whitelist(args: TrainingArgs) -> List[str]:
    """配置驱动的过滤行为：白名单字符串 -> 列表（组合了上面的 parse_csv）。"""
    return parse_csv(args.defect_whitelist)


# ---------------------------------------------------------------------------
# 演示入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("1) 类型标注函数：CSV 解析")
    print("-" * 60)
    print(f"parse_csv('砂眼, 崩边')      -> {parse_csv('砂眼, 崩边')}")
    print(f"parse_csv(None)              -> {parse_csv(None)}")
    print(f"parse_csv_int('1, 2')        -> {parse_csv_int('1, 2')}")

    print("=" * 60)
    print("2) 类：GrayImage")
    print("-" * 60)
    img = GrayImage([[10, 200], [30, 240]])
    print(f"img                        -> {img}")
    print(f"img.binarize(threshold=100)-> {img.binarize(100).tolist()}")

    print("=" * 60)
    print("3) 装饰器")
    print("-" * 60)

    @count_calls
    def fake_train_step(step: int) -> int:
        return step + 1

    for _ in range(3):
        fake_train_step(0)

    import time
    t0 = time.perf_counter()
    slow_square(9)
    slow_square(9)          # 第二次应命中缓存，几乎不耗时
    slow_square(9)
    elapsed = time.perf_counter() - t0
    print(f"    [memoize] 3 次 slow_square(9) 共耗时 {elapsed*1000:.1f} ms "
          f"（无缓存应约 30ms，缓存命中应约 10ms）")
    print(f"    [memoize] 缓存内容: {slow_square.cache}")

    print("=" * 60)
    print("4) dataclass：配置")
    print("-" * 60)
    args = TrainingArgs(train_mode="finetune", epochs=5, lr=1e-4)
    print(f"args.summary() -> {args.summary()}")
    print(f"apply_whitelist -> {apply_whitelist(args)}")
