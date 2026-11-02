# Stage 3：语义分割学习工程（4~6 周）

为读懂 `supervised_training/` 全部（模型/损失/评估）而设。前置：完成
`../stage2_pytorch/`。

## 工程结构

```
stage3_segmentation/
├── stage3/
│   ├── encoder_decoder.py         级联头形状链(64→128→256)+双线性软边 vs 反卷积
│   ├── unet_mini.py               迷你 U-Net：跳连为什么救细节(有对照实验)
│   ├── losses_focal_dice.py       ★Demo F：手写 Focal/Dice+1% 前景不平衡实验
│   ├── pointrend_sampling.py      抽题官复刻：3 倍候选/75:25/65,536 点
│   ├── threshold_sweep.py         多阈值 Micro 指标+best-F1+两把尺差异
│   ├── demo_g_dataset.py          ★Demo G 骨架：SynthNEU 生成器+NEU-DET 目录契约
│   └── demo_h_sam_prompt.py       ★Demo H 概念版：promptable vs 闭集二分类
├── tests/                         pytest 用例
├── conftest.py / requirements.txt
└── README.md
```

## 七个文件大白话导读（也是阅读顺序）

| 文件 | 大白话讲一件什么事 |
|------|-------------------|
| `encoder_decoder.py` | 分割网的标准体型：先**压缩提炼**（编码器，骨干那套）再**放大还原**成逐像素答案（解码器）；对比"双线性软边 vs 反卷积"两种放大画法——边缘怎么画回来 |
| `unet_mini.py` | 压缩会丢细节（缺陷那么小，丢不起）——U-Net 的**跳连**把压缩时丢的细节抄送回来；带"有无跳连"对照实验 |
| `losses_focal_dice.py` ★ | 核心难点：缺陷常只占图面 **1%**、99% 是背景，普通损失会让模型偷懒"全答背景"。手写 **Focal**（难题多罚）+ **Dice**（按面积打分）+ 1% 前景实验——对应主仓库总分公式 `focal + 3*dice` |
| `pointrend_sampling.py` | **抽题官**：一张图上百万像素大多是"闭眼会答"的背景题——只挑最有疑问的约 6%（65,536 取 3 倍候选、75:25 混简单题）算损失，省算力不偏科 |
| `threshold_sweep.py` | 模型输出 0~1 的"疑心值"，**几分以上算缺陷**？扫一排阈值各算指标找 best-F1；对比 Micro 平均 vs 逐样本平均两把尺何时打架 |
| `demo_g_dataset.py` ★ | 数据骨架：SynthNEU 生成器 + NEU-DET（真实钢材缺陷数据集）目录契约——从合成小图通向真业务数据的桥 |
| `demo_h_sam_prompt.py` ★ | 概念对比：SAM 的"**你圈哪我分割哪**"（可提示、开集）vs 本工程"固定认这几种缺陷"（闭集二分类）——业界两条路线 |

与 stage2 的关系一句话：训练机器（循环/调度/断点）全盘复用，只换三样——模型的头（每图打分 → 输出一张答案图）、损失（交叉熵 → Focal+Dice）、评估（命中率 → IoU/逐像素指标）。

## 环境与运行

```bash
cd D:/ytc_robot/vision_learn/stage3_segmentation
# 复用 stage2 的 .venv（同依赖）或自建
.venv/Scripts/python.exe -m stage3.encoder_decoder
.venv/Scripts/python.exe -m stage3.unet_mini
.venv/Scripts/python.exe -m stage3.losses_focal_dice
.venv/Scripts/python.exe -m stage3.pointrend_sampling
.venv/Scripts/python.exe -m stage3.threshold_sweep
.venv/Scripts/python.exe -m stage3.demo_g_dataset
.venv/Scripts/python.exe -m stage3.demo_h_sam_prompt

.venv/Scripts/python.exe -m pytest tests/ -v
```

## 30 天学习日程（每天 ~1.5 小时）

| 天 | 内容 | 对应主仓库 |
|----|------|-----------|
| 1-2 | `encoder_decoder`：形状链/软边；论文 U-Net | `CascadedSegHead` 64→128→256→1024 |
| 3-5 | **作业 Demo E**：克隆 `milesial/Pytorch-UNet` 跑通 Oxford-IIIT Pet | 级联头是它的"预算版" |
| 6-7 | `unet_mini`：跳连对照实验亲手跑；读论文 V-Net(Dice 起源) | — |
| 8-11 | ★Demo F：手写三损失→对公式→1% 前景实验；读论文 Focal(RetinaNet) | `loss_utils.py` 全部数字 |
| 12-13 | `pointrend_sampling`：抽题官+为什么只批 6%；读论文 PointRend | `loss_masks` 65536/3.0/0.75 |
| 14-15 | `threshold_sweep`：Micro vs 逐样本两把尺 | `eval.py run_sweep` |
| 16-17 | Transformer 骨干概念(ViT)+ 轻量化(EfficientViT)：看 B站霹雳吧啦Wz 分割篇补课 | `seg.py` 骨干来自 vision_evs |
| 18-19 | **作业 Demo H**：跑 `facebookresearch/segment-anything`；对照 `demo_h_sam_prompt` 概念版 | 本工程=无 prompt 闭集分割 |
| 20-23 | **作业 Demo G**：下载 NEU-DET(或 DAGM)，归整成目录契约，用 Demo F 的损失训自己的小分割网 | 数据拓扑映射 |
| 24-25 | 进阶：Kaggle Severstal 钢材缺陷赛金银方案解读 | 业务同类 |
| 26-28 | 回读主仓库 `core/seg.py` + `core/loss_utils.py` 全文 | — |
| 29-30 | 回读 `eval.py` 阈值扫描分支 | — |

## 通关标准（全部达成即完成阶段 3）

- [ ] 说清"为什么这个工程用 Focal+Dice 而不用纯 BCE"（偷懒解/难像素/面积三视角）
- [ ] 说清"PointRend 采 256×256×3 个点是在省什么"（百万像素 vs 6% 简单题）
- [ ] 双线性软边 vs 反卷积可学边的对比结论（软边占比有测试）
- [ ] 跳连为什么救细节（U-Net vs 无跳连 Dice 差有测试）
- [ ] Micro 平均与逐样本平均何时给出不同结论（构造案例在测试里）
- [ ] Demo E 跑通 U-Net；Demo G 在 NEU-DET 上训出自己的模型（IoU 自报）
- [ ] `pytest tests/` 全绿

## 与主仓库的参数级对应

| 本工程 | 主仓库 |
|--------|--------|
| `CascadeHead` 形状链 64→128→256 | `core/seg.py` 前两级逐级同构 |
| `sigmoid_focal_manual(α=0.4, γ=2)` | `sigmoid_focal_loss` 同参 |
| `dice_loss_manual(smooth=1)` | `dice_loss` +1 平滑 |
| 组合 `focal + 3*dice` | `trainer.train` 127 行总分公式 |
| `sample_uncertain_points(65536, 3.0, 0.75)` | `loss_masks` 三个数字 |
| `uncertainty=-|logit|` | `calculate_uncertain` |
| `sweep_metrics` + eps=1e-7 | `run_sweep` Micro 指标 |
| 掩码永远 NEAREST（软边测试） | 2.3 预处理口径 |

## 论文清单（按顺序）

1. U-Net（2015）——跳连与编解码
2. Focal Loss / RetinaNet（2017）——类不平衡
3. V-Net（2016）——Dice 损失起源
4. PointRend（2020）——不确定性点采样

## 下一步

进入阶段 4（无监督异常检测）：`../stage4_anomaly/`。
