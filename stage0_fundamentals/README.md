# Stage 0：前置基础学习工程（1~2 周）

为读懂 `D:\ytc_robot\vision-ai-training`（工业视觉质检 AI 训练平台）而设的前置课程。
**每个知识点都标注了它对应主工程仓库的哪段代码**——学完这里，回去读仓库就没有语言和数学障碍。

## 工程结构

```
stage0_fundamentals/
├── stage0/                    # 4 个示例模块（可运行演示 + 被测试的实现）
│   ├── py_basics.py           # Python：函数/类/装饰器/类型标注/dataclass 配置
│   ├── numpy_basics.py        # NumPy：创建/切片/广播/归约/dtype 溢出/视图
│   ├── math_basics.py         # 数学：矩阵乘/向量距离/方差公式/分位数/AUC
│   └── image_as_array.py      # 图像即数组：RGB 形状/灰度化/手写 3x3 均值滤波
├── tests/                     # 59 个 pytest 用例（独立参考实现交叉验证）
├── conftest.py
├── requirements.txt           # numpy + pytest（仅此两个）
└── README.md
```

## 环境搭建（Windows Git Bash）

```bash
cd D:/ytc_robot/vision_learn/stage0_fundamentals
python -m venv .venv                                   # 已创建可跳过
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

日常激活（可选，直接用完整路径调 python 也行）：

```bash
source .venv/Scripts/activate
```

## 怎么用（核心学习循环）

每个模块有**演示入口**和**测试**两套用法，建议循环：

1. **跑演示**，对着打印输出和源码注释理解"每行为什么"；
2. **改参数再跑**（比如把 3x3 滤波改成 5x5、噪声比例调大），预测结果再验证；
3. **跑测试**，读测试里的断言——它们是"这个概念的正确性标准"；
4. **合上书改写**：把某个手写函数删掉自己重写，跑测试验证（红→绿循环）。

```bash
# 演示（推荐顺序）
.venv/Scripts/python.exe -m stage0.py_basics
.venv/Scripts/python.exe -m stage0.numpy_basics
.venv/Scripts/python.exe -m stage0.math_basics
.venv/Scripts/python.exe -m stage0.image_as_array

# 测试（全部应通过：59 passed）
.venv/Scripts/python.exe -m pytest tests/ -v
```

## 10 天学习日程（每天 ~1.5 小时）

| 天 | 内容 | 对应主仓库的哪里 |
|----|------|----------------|
| 1-2 | `py_basics.py`：类型标注、类与 property、装饰器（计数/缓存）、**dataclass 配置** | `confs/training_config.py` 的 TrainingArgs、`train_entry.py` 的 `_parse_csv` |
| 3-4 | `numpy_basics.py`：向量化思维、切片裁图、axis 归约、**uint8 溢出陷阱**、视图 vs 拷贝 | patch 切片 `(x0,y0,patch_size)`、ImageNet 归一化、亮度扰动防溢出 |
| 5 | `math_basics.py` 前半：手写矩阵乘、欧氏距离、**逐位置通道距离（迷你异常图）** | 无监督线 `predict()`: `mean((teacher-student)², dim=通道)` |
| 6 | `math_basics.py` 后半：`E[x²]-E[x]²` 方差、**kthvalue 口径分位数**、困难挖掘、手写 AUC | `teacher_normalization`、`_map_quantiles`(90%/99.5%)、`trainer.py`(前 0.1%)、AUC 门控保存 |
| 7-8 | `image_as_array.py`：RGB=数组、两种灰度化、**手写 3x3 均值滤波（零填充+滑窗）** | 卷积层的雏形、eval 里的 GaussianBlur |
| 9 | 滤波应用实验：去椒盐噪声（误差怎么算）、局部对比度伪异常检测、**零填充边界效应** | region 门控排除边界假阳的直觉来源 |
| 10 | 总复习：通关自测四题（见 `image_as_array.py` 末尾）+ 合书重写 `mean_filter_3x3_by_hand` 和 `kth_smallest`，测试通过即毕业 |

## 通关标准（全部达成即完成阶段 0）

- [ ] 一句话说清：**一张 RGB 图在内存里就是 (H, W, 3) 的 uint8 数组**（1024×1024 RGB 占 3 MB）
- [ ] 不查资料写出 `image[y0:y0+size, x0:x0+size]` 的裁图切片（y=行/高在前）
- [ ] 解释 uint8 溢出：`(250*1.15).astype(uint8)` 为什么得 31 而不是 255
- [ ] 解释 `axis` 归约：`(H,W,3).mean(axis=(0,1))` 得到什么、为什么
- [ ] 手写矩阵乘和欧氏距离，并知道它对应卷积/异常分数
- [ ] 用 `E[x²]-E[x]²` 算方差，说明为什么要 `max(var, 0)` 钳制
- [ ] 复述仓库分位数口径 `k = clamp(int(n*q), 1, n)` 取第 k 小，以及 90%/99.5%/99.9% 三个分位各自用在无监督线的哪个环节
- [ ] 手写零填充 3x3 均值滤波（合书），并解释它为什么去噪、为什么变糊、边界为什么不守恒
- [ ] `pytest tests/` 全绿

## 测试设计说明

测试不用"自己测自己"：手写实现 vs NumPy 的**另一个算法**交叉验证——

| 手写实现 | 独立参考 |
|----------|----------|
| `matmul_by_hand`（三重循环） | `np.matmul`（BLAS） |
| `l2_distance_by_hand` | `np.linalg.norm` |
| `kth_smallest`（排序取第 k） | `np.partition`（quickselect） |
| `mean_filter_3x3_by_hand`（滑窗循环） | `sliding_window_view().mean()`（视图向量化） |
| `auc_by_hand`（两两比较定义式） | `sklearn.roc_auc_score`（若装了 sklearn） |

这也是工程实践的常识：**新实现要与已知正确的异构实现对拍**。

## 常见坑（本工程开发时真实踩过，测试里都钉了断言）

- `uint8` 直接乘浮点再转回 → 溢出回绕（250 变 31）；
- 切片是**视图**：改裁出的 patch 会改到原图，要隔离必须 `.copy()`；
- 零填充均值滤波：亮图最外圈会被压暗——统计/检测要排除边界一圈；
- 高频随机纹理不适合验证"去噪"：滤波会把自然纹理一起抹掉（误差反而变大）；
- `E[x²]-E[x]²` 浮点残渣：可能出负数 → 开方 NaN，必须钳 0（仓库同款防御）。

## 下一步

进入阶段 1（传统图像处理/OpenCV，3~4 周）：JIT 标件配准、形态学、连通域。
届时在 `vision_learn/` 下新建 `stage1_opencv/` 子工程继续。
