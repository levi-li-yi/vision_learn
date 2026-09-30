"""Python 语言要点：函数 / 类 / 装饰器 / 类型标注 / dataclass 配置风格。

为什么学这些：vision-ai-training 的全部代码风格 = 类型标注函数 + dataclass 配置
（confs/training_config.py 的 TrainingArgs 就是 @dataclass），CLI 参数从 dataclass
自动生成。把本文件的每个概念吃透，回去读仓库的 confs/ 和 base_utils/parse_utils.py
就不会有语言层面的障碍。

读装饰器前的前置认知（一句话）：Python 里函数是"物件"（像 U 盘）——
可以赋给变量、当参数传给别的函数、被 return 出来、还能挂属性。
装饰器全部建立在这个基础上，没有这个认知它就是天书。

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
    # 列表推导式一行干三件事：split 按逗号拆段 -> strip 去空白并兼任过滤条件
    # （空字符串判定为"假"，空段被丢弃）-> 开头的 strip 是收进列表的干净值。
    # strip 出现两次因为在两个位置干两件事：当条件用、当元素值用。
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_csv_int(raw: Optional[str]) -> List[int]:
    """逗号分隔整数解析（对应仓库 --region_whitelist "1,2"）。"""
    return [int(item) for item in parse_csv(raw)]


# ---------------------------------------------------------------------------
# 2. 类：封装"图"这个概念（迷你版，为阶段 1 的真实图像处理热身）
# ---------------------------------------------------------------------------

class GrayImage:
    """一张灰度图 = 二维 uint8 数组。

    大白话：类 = 图纸，对象 = 按图纸造出的实物（执行 img = GrayImage(...)
    那一刻才造出实物，class 定义本身只是图纸）。
    封装的意义：把"数据 + 对数据合法的操作"打包，出生即校验——
    裸数组谁都能塞非法东西，对象天生保证自己合法。

    本类四个角色：出生(__init__) -> 查询(@property) -> 操作(方法) -> 打印(__repr__)。
    """

    def __init__(self, pixels):
        """构造函数：造实物那一刻 Python 自动调用（永远不用自己调）。

        self = "当前这个实物自己"。img = GrayImage(x) 在 Python 眼里是
        GrayImage.__init__(img, x)——第一个参数是自动传进去的：
        谁调用，self 就是谁（img1 上调用 self 就是 img1，互不干扰）。
        """
        # 不管调用者传列表还是数组，统一转成 ndarray（已是数组则不复制），
        # 后面才能用 .ndim/.min() 这些数组才有的能力
        pixels = np.asarray(pixels)

        # 构造即校验（防御式编程）：错误越早暴露越好定位——
        # 出生时拦下非法输入，好过三天后在某个深层函数里以诡异方式崩溃。
        # 仓库里"先检查、不合法就 raise/降级"的写法到处都是。
        if pixels.ndim != 2:
            raise ValueError(f"灰度图必须是二维数组，收到 ndim={pixels.ndim}")
        if pixels.min() < 0 or pixels.max() > 255:
            raise ValueError("像素值必须落在 [0, 255]")

        # 实例属性：挂在实物身上的数据，类里所有方法都经 self._pixels 访问。
        # uint8 = 无符号 8 位整数（0~255），图像的标准容器。
        # 名字开头的下划线 = "内部私有，外面别直接摸"——Python 的君子协定，
        # 不强制但约定俗成；外部想拿信息走下面的查询接口。
        self._pixels = pixels.astype(np.uint8)

    # @property 的效果：调用时不加括号（img.shape 不是 img.shape()）。
    # 分工记忆法：查询用 property（问事实，不加括号），操作用方法（做动作，加括号）。
    @property
    def shape(self):
        """(高, 宽)——注意顺序是 行在前。

        shape 不是本类定义的东西：_pixels 是 NumPy 数组，.shape/.ndim/.dtype
        都是 ndarray 出厂自带的属性，这里只是转手。
        点链读法从左往右：self._pixels 先拿到数组对象，再取它的 .shape。
        """
        return self._pixels.shape

    @property
    def mean_brightness(self):
        """平均亮度。为什么做成 property 而不是存个值？因为它是"现场算"的——
        属性存不下随数据变化的结果，查询接口才算是活的。"""
        return float(self._pixels.mean())

    def binarize(self, threshold: int):
        """二值化：大于阈值为 1，否则为 0——工业质检"判好判坏"的最底层操作。

        为什么是方法不是 property：需要外部参数（阈值是外部决策，不是图自带的）。
        (self._pixels > threshold) 是 NumPy 广播：一个数和整个数组比 = 逐元素比，
        得到布尔数组；.astype(uint8) 把 True/False 变 1/0。
        仓库 eval 里 (probs > threshold) 是同一个动作。

        无副作用风格：返回新数组，不修改 self._pixels——原 img 完好无损。
        """
        return (self._pixels > threshold).astype(np.uint8)

    def __repr__(self):
        """教 Python 怎么打印自己：不定义的话 print(img) 输出的是
        <...object at 0x...> 一串没用的内存地址；定义后一眼看到关键信息。
        注意内部复用 self.shape / self.mean_brightness——好代码吃自己的接口。"""
        return f"GrayImage(shape={self.shape}, mean={self.mean_brightness:.1f})"


# ---------------------------------------------------------------------------
# 3. 装饰器：不改动原函数就给它加能力
# ---------------------------------------------------------------------------

def count_calls(func: Callable):
    """调用计数装饰器：训练日志里"这是第几个 step"就是类似机制。

    @count_calls 是语法糖，完全等价于：
        def fake_train_step(...): ...
        fake_train_step = count_calls(fake_train_step)
    一句话：把下面的函数送去加工厂，成品继续用原名。
    所以装饰器本体只是个"吃函数、吐函数"的普通函数。

    两个时间点分开想（理解装饰器的钥匙）：
      装饰时（@ 那一刻，只发生一次）：造计数器、定义 wrapper（定义不等于执行，
        函数体一行都没跑）、返回 wrapper 顶替原名——从此全世界调用
        fake_train_step，进门的都是 wrapper，原函数被藏在 wrapper 肚子里
        的 func 名下（装饰后 __name__ 变成 'wrapper' 就是证据）；
      调用时（每次）：计数 +1 -> 调原函数 -> 汇报 -> 把结果原样转交。
    """
    # 计数器为什么用 dict 不用普通数字？Python 的规矩：函数内部只要出现
    # 赋值号（calls += 1 本质是赋值），这个名字就成了函数私有变量，和外面
    # 断了联系还报 UnboundLocalError——"换名字会断亲"。
    # 字典写法 calls["n"] += 1 是"改本子内容"，不碰赋值红线，永远翻的是
    # 外层这本账——"改内容不断亲"。
    # 这就是闭包：wrapper 带走了外层函数的变量，外层函数虽已返回，账本还活着。
    calls = {"n": 0}

    def wrapper(*args, **kwargs):
        # *args 收拢所有位置参数、**kwargs 收拢关键字参数，到下一行再原样
        # 抖开转发——快递中转站：不管包裹内容，原样转寄，使 wrapper 能包任何函数
        calls["n"] += 1
        result = func(*args, **kwargs)
        print(f"    [count_calls] {func.__name__} 第 {calls['n']} 次调用完成")
        return result

    return wrapper


def memoize(func: Callable):
    """记忆化装饰器：缓存已算过的结果（一本"记事本"）。

    调用时只有两条路：
      未命中：args 不在 cache -> 真算（付成本）-> 结果记到本上 -> 返回；
      命中：args 在 cache -> 跳过 func 一步不算（免费）-> 直接取结果。
    （演示里 3 次 slow_square(9) 只花第 1 次的 sleep 时间。）

    对应仓库的真实语义：无监督线 teacher 权重冻结，同一张图输出永远
    相同——"确定性输出才配缓存"是同一个设计直觉。
    """
    # 记事本：键用参数元组 args——元组天生不可变（可哈希），且"参数组合"
    # 正好做查询键：同样 9 来了命中，换 10 就是新键 (10,)，各记各的
    cache = {}

    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]

    # 把记事本挂在替身身上当属性（函数是物件，能挂东西）。
    # 纯粹是开观察窗让外面偷看（测试靠 slow_square.cache 验证"只算了一次"），
    # 删掉这行功能完全不变。
    wrapper.cache = cache
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
    extras: List[str] = field(default_factory=list)  # 可变默认值（列表）必须用
                                                    # field(default_factory=...)，
                                                    # 否则所有实例共享同一个列表

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
    print(f"img                        -> {img}")   # 触发 __repr__
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
    slow_square(9)          # 第 1 次：未命中，真算（约 10ms）
    slow_square(9)          # 第 2、3 次：命中记事本，跳过 sleep
    slow_square(9)
    elapsed = time.perf_counter() - t0
    print(f"    [memoize] 3 次 slow_square(9) 共耗时 {elapsed*1000:.1f} ms "
          f"（无缓存应约 30ms，缓存命中应约 10ms）")
    print(f"    [memoize] 缓存内容: {slow_square.cache}")   # 观察窗：{(9,): 81}

    print("=" * 60)
    print("4) dataclass：配置")
    print("-" * 60)
    args = TrainingArgs(train_mode="finetune", epochs=5, lr=1e-4)
    print(f"args.summary() -> {args.summary()}")
    print(f"apply_whitelist -> {apply_whitelist(args)}")
