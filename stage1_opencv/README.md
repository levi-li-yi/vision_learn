# Stage 1：传统图像处理 / OpenCV 学习工程（3~4 周）

为读懂 `vision-ai-training` 的**配准、门控、合成底座**而设。前置：完成
`../stage0_fundamentals/`（NumPy 基础与"图像即数组"）。

## 工程结构

```
stage1_opencv/
├── stage1/
│   ├── io_crop_filter.py        读写图/BGR陷阱/裁剪/高斯与中值滤波（手写对拍）
│   ├── threshold_morphology.py  二值化（固定/Otsu）+ 形态学（腐蚀/膨胀/开闭）
│   ├── connected_components.py  连通域统计 + 外轮廓（eval 的组件 bbox+面积）
│   ├── transforms.py            仿射/透视变换（dsize 顺序陷阱）
│   ├── registration.py          ★Demo A：ORB+RANSAC vs 相位相关 + 混合配准
│   ├── blending.py              Alpha 混合/羽化/贴纸变换（Cut-Paste 迷你版）
│   └── defect_pipeline.py       ★Demo B：经典质检流水线 + region 门控
├── tests/                       74 个 pytest 用例
├── conftest.py / requirements.txt（numpy + pytest + opencv-python）
└── README.md
```

★ = 提纲点名的两个 Demo，`registration.py` 同时是本阶段的**通关模块**。

## 环境与运行

```bash
cd D:/ytc_robot/vision_learn/stage1_opencv
python -m venv .venv                      # 已创建可跳过
.venv/Scripts/python.exe -m pip install -r requirements.txt

# 演示（建议按文件名顺序）
.venv/Scripts/python.exe -m stage1.io_crop_filter
.venv/Scripts/python.exe -m stage1.threshold_morphology
.venv/Scripts/python.exe -m stage1.connected_components
.venv/Scripts/python.exe -m stage1.transforms
.venv/Scripts/python.exe -m stage1.registration
.venv/Scripts/python.exe -m stage1.blending
.venv/Scripts/python.exe -m stage1.defect_pipeline

# 测试（应 74 passed）
.venv/Scripts/python.exe -m pytest tests/ -v
```

## 20 天学习日程（每天 ~1.5 小时）

| 天 | 内容 | 对应主仓库 |
|----|------|-----------|
| 1-2 | `io_crop_filter`：BGR 陷阱、imwrite/imread 往返、切片裁剪 | patch 切片 `(x0,y0,patch_size)` |
| 3-4 | 高斯核公式与可分离性、手写高斯/中值 vs cv2 对拍 | 无监督 eval 的 `GaussianBlur(33,33,σ4)` |
| 5-6 | `threshold_morphology`：固定阈值→Otsu 的动因 | 概率图 `> threshold` 二值化 |
| 7-8 | 形态学四件套；"结构元要比洞大"的口径 | eval 的 `MORPH_CLOSE(15,15)` 椭圆核 |
| 9-10 | `connected_components`：stats/质心/面积门限 | eval 组件统计表 + `--min_area` |
| 11-12 | `transforms`：2x3 矩阵、dsize=(w,h)、点位变换 | 配准后 warp 蒙版 |
| 13-14 | 透视与四点拉正 | 斜拍矫正的通用工具 |
| 15-17 | ★`registration`：ORB→比率测试→RANSAC→内点比门控；相位相关；混合策略；`warp_mask` | `data_utils/data.py` JIT 混合配准（内点比 0.3 / align_min_score / align_max_shift 全部同参数） |
| 18 | `blending`：羽化 alpha、贴纸变换、GT 收敛 | DefectTransformer / DefectBlender / `gt &= allowed` |
| 19-20 | ★`defect_pipeline`：经典质检 + 门控；总结规则方法的力与极限 | region AND 门控 |
| 附加 | 合书重写 `hybrid_register` + `warp_mask`，测试全绿即毕业 | — |

## 通关标准（全部达成即完成阶段 1）

- [ ] 不查资料说出 BGR 陷阱和 dsize=(w,h) 顺序（两处各踩过一次测试）
- [ ] 解释"中值去椒盐、高斯去高斯噪声"的机理差异
- [ ] 解释 Otsu 的适用前提（双峰），以及单峰背景下的病理行为（有测试钉住）
- [ ] 说清腐蚀/膨胀与开/闭的一对一关系（对偶性测试）
- [ ] **给出标件图和现场图，独立写出"估相似变换 -> warp 蒙版到现场坐标系"脚本**（`registration.py` 就是答案，端到端 IoU>0.95 有测试）
- [ ] 复述两种配准器的失效区与互补逻辑：ORB 无角点即死、相位相关见旋转即废
- [ ] 解释 region 门控的业务含义与 Cut-Paste 羽化的"合成域差"动机
- [ ] `pytest tests/` 全绿

## 与主仓库的参数级对应（读仓库时的锚点）

| 本工程 | 主仓库 |
|--------|--------|
| `hybrid_register(min_inlier_ratio=0.3)` | ORB 内点比阈值 0.3 |
| `hybrid_register(min_phase_response=0.35)` | `align_min_score=0.35` |
| `hybrid_register(max_shift=64)` | `align_max_shift=64` |
| `hybrid_register` 返回 `"none"` 降级 | 配准失败退化原始 region + `region_margin` 安全带 |
| `warp_mask` 用 INTER_NEAREST | region 蒙版 `>0` 二值语义 |
| `cut_paste(allowed=...)` 的 `gt &= allowed` | 防御性 GT 收敛 |
| `classical_detect` 的 min_area | eval `--min_area`（默认 100px） |
| `region_gate` | 推理 FP 空间门控 AND |

## 常见坑（开发本工程时真实踩过，测试里都有断言）

- `cv2.GaussianBlur` 对 **uint8** 输入走定点快速实现，与手写浮点卷积差 ~1 灰度级（float32 输入则逐位一致）——对拍时选对 dtype；
- `warpAffine` 的 `dsize=(宽, 高)`，传成 `(高, 宽)` 图会被转置裁切（旋转演示踩过：500px 白块变 350px）；
- 连通域编号按**光栅扫描顺序**，别假设 `stats[1]` 是哪个目标——按面积/位置定位；
- 圆的像素数 ≠ `πr²`（离散化），测试真值要么用输入侧统计、要么容差；
- 形态学闭运算**结构元要比洞大**才填得上（5x5 洞 3x3 核填不动，有正反两个测试）；
- Otsu 对单峰背景会切在噪声中间，拉丝纹理整行误检（"规则方法极限"的微观标本）。

## 下一步

进入阶段 2（PyTorch 与深度学习骨架，4~6 周）：训练循环/调度器/迁移学习/DDP。
届时在 `vision_learn/` 下新建 `stage2_pytorch/`。
