# Stage 2：PyTorch 与深度学习入门学习工程（4~6 周）

为读懂 `vision-ai-training/` 两个 `trainer.py` 的全集而设：DDP+SyncBN+
BF16+梯度累积+双组学习率+三模式断点。前置：完成 `../stage0_fundamentals/`
（NumPy/图像即数组）。

## 工程结构

```
stage2_pytorch/
├── stage2/
│   ├── tensor_autograd.py         Tensor/自动求导/三种浮点位分配(bf16 为什么不溢出)
│   ├── nn_module_dataset_loader.py nn.Module/Dataset/DataLoader 微缩三件套
│   ├── train_loop.py              完整训练循环+梯度累积(等价性有测试钉死)
│   ├── optim_scheduler.py         AdamW 双组旋钮+热身→余弦(断言/曲线全复刻)
│   ├── amp_scaler.py              autocast bf16+GradScaler+fp16 溢出复现
│   ├── checkpoint_modes.py        断点五件套+fresh/resume/finetune
│   ├── demo_c_cifar.py            ★Demo C：CIFAR-10 完整训练脚本(smoke 离线可测)
│   └── demo_d_transfer.py         ★Demo D：迁移学习冻结 vs 差异化学习率
├── tests/                         pytest 用例
├── conftest.py / requirements.txt（torch+torchvision+numpy+pytest）
└── README.md
```

## 环境与运行

```bash
cd D:/ytc_robot/vision_learn/stage2_pytorch
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
# torch CUDA 版：pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

.venv/Scripts/python.exe -m stage2.tensor_autograd
.venv/Scripts/python.exe -m stage2.nn_module_dataset_loader
.venv/Scripts/python.exe -m stage2.train_loop
.venv/Scripts/python.exe -m stage2.optim_scheduler
.venv/Scripts/python.exe -m stage2.amp_scaler
.venv/Scripts/python.exe -m stage2.checkpoint_modes
.venv/Scripts/python.exe -m stage2.demo_c_cifar            # smoke 自检
.venv/Scripts/python.exe -m stage2.demo_d_transfer

.venv/Scripts/python.exe -m pytest tests/ -v
```

## 30 天学习日程（每天 ~1.5 小时）

| 天 | 内容 | 对应主仓库 |
|----|------|-----------|
| 1-2 | `tensor_autograd`：手算 vs autograd、三种浮点位分配表 | 4.5 bf16/SDPA 替身、fp16 事故 |
| 3-4 | PyTorch 官方 60-Minute Blitz（配合小土堆视频查漏） | — |
| 5-6 | `nn_module_dataset_loader`：模块/数据集契约/装桶 | `data.py` 契约、`allocate` |
| 7-8 | `train_loop`：单 epoch 循环；**累积等价性测试亲手跑** | `trainer.train` 127-141 |
| 9-10 | `optim_scheduler`：两组旋钮+断言+调度三部曲；默画 lr 曲线 | `trainer.__init__` 39-88 |
| 11-12 | `amp_scaler`：bf16/fp16 位级差异、GradScaler 三步曲 | `trainer` 118-121、4.5 案例 |
| 13-14 | `checkpoint_modes`：五件套、三模式；对照双文件生命周期图 | `base_utils/utils.py` |
| 15-17 | ★Demo C smoke 跑通→改真实 CIFAR-10（30 epochs 冲 85%） | 全流程骨架 |
| 18-19 | ★Demo D：冻结 vs 差异化两组准确率对比 | `backbone_lr_scale=0.1` |
| 20-21 | DDP 概念（B站 RethinkFun 动画）+ 单机改 DDP 要点视频；本机单卡看懂 `mp.spawn` 分支即可 | `train_entry.run` |
| 22-24 | d2l 前半本补理论（优化器/正则/数值稳定性章节优先） | — |
| 25-27 | 合书**默写**带 AMP+调度器+续训的训练循环（Demo C 关掉对着写） | — |
| 28-30 | 回读主仓库 `supervised_training/trainer.py` 全文——此时应"不过如此" | — |

## 通关标准（全部达成即完成阶段 2）

- [ ] 不查资料默写：autocast 前向→loss÷累积→(中间步 no_sync / 最后一步五连击)→记账
- [ ] 解释 warmup 保护预训练权重的机理（起步旋钮乱+梯度噪声大 -> 全速拧坏老底子）
- [ ] 解释 bf16 与 fp16 的位差异及"天然不溢出"（测试有位分配断言）
- [ ] 证明梯度累积等价性（`tests/test_train_loop.py` 的数学依据说一遍）
- [ ] 说出断点五件套是什么、三模式各恢复哪些
- [ ] Demo C 真实 CIFAR-10 test acc ≥ 85%（或说明差距原因）
- [ ] `pytest tests/` 全绿

## 与主仓库的参数级对应

| 本工程 | 主仓库 |
|--------|--------|
| `split_param_groups(lr, scale=0.1)` | `trainer.py` 46-53 两组旋钮 |
| `assert_backbone_floor` | 39-44 行 AssertionError |
| `build_scheduler(warmup=2000)` | LinearLR→SequentialLR→Cosine 同结构 |
| `train_one_epoch(accumulation_steps)` | 122-141 损失÷N+条件五连击 |
| `save_five/load_five` | `save_trainer_state/load_trainer_state` |
| `apply_mode` 三分支 | fresh 删断点 / resume 全恢复 / finetune 只穿权重 |
| Demo D `split_transfer_groups` | `backbone_lr_scale=0.1` 的用意 |

## 常见坑（测试里都有断言）

- 调度器 `step()` 按**优化器步**计：开累积时总步数要 `steps // accumulation_steps`；
- `optim.zero_grad()` 只在累积最后一步调用，中间清零=白攒；
- GradScaler 配 bf16 近似恒等——它真正救的是 fp16 下溢（实验里有 1e-8 案例）；
- fp16 上限 65504：`1e5` 直接 inf（无监督线推理事故的 10 行复现）；
- `load_state_dict(strict=False)` 是断点兼容的宽容位，新键旧键都能活。

## 下一步

进入阶段 3（语义分割）：`../stage3_segmentation/`。
