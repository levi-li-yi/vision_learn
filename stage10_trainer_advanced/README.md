# Stage 10：训练器进阶学习工程（2~3 周）

为读懂 `supervised_apps/train/src/train/seg.py + trainer.py + main.py(训练段) +
ckpt_utils.py`（Conv-LoRA 注入、EMA 影子权重、四参数组、DDP 多卡、GPU 侧指标）
而设。前置：完成 `../stage2_pytorch/`（训练循环/调度器/AMP）与 `../stage8/`
（Linear-LoRA）；本阶段把 LoRA 搬进 Conv 骨干并补齐训练器工程件。

## 工程结构

```
stage10_trainer_advanced/
├── stage10/
│   ├── conv_lora.py      Conv-LoRA 原位注入 / partial-vs-闭包 deepcopy 陷阱 / BN 断言 / 起步权重识别
│   ├── ema_groups.py     EMA(热身/buffer 分类/_foreach) / 四参数组 AdamW / autotune
│   ├── gpu_metrics.py    GPU 侧指标累加 pop_metrics / batch IoU / 考试算型暖机
│   └── ddp_spawn.py      mp.spawn + gloo 双进程真实 DDP / 广播 / all_reduce 参数一致
├── tests/                pytest 用例（含真实双进程 DDP）
├── conftest.py / requirements.txt（torch + numpy + pytest）
└── README.md
```

## 环境与运行

```bash
cd D:/ytc_robot/vision_learn/stage10_trainer_advanced
python -m venv .venv                      # 或复用 stage2 的 venv（同依赖）
.venv/Scripts/python.exe -m pip install -r requirements.txt

.venv/Scripts/python.exe -m stage10.conv_lora
.venv/Scripts/python.exe -m stage10.ema_groups
.venv/Scripts/python.exe -m stage10.gpu_metrics
.venv/Scripts/python.exe -m stage10.ddp_spawn     # 真实双进程 DDP (gloo/CPU)

.venv/Scripts/python.exe -m pytest tests/ -v
```

## 15 天学习日程（每天 ~1.5 小时）

| 天 | 内容 | 对应主仓库 |
|----|------|-----------|
| 1 | `conv_lora`：恒等起步/键名不变/冻结；改 r=4/64 看挂载层数 | `seg.py::_attach_lora` |
| 2-3 | **deepcopy 陷阱**：跑 partial/闭包对照实验，读懂 seg.py 121~128 行注释 | 同上 |
| 4 | BN 断言体检：注释掉 train() 覆盖看冻结段 BN 被偷改 | `trainer.py`(_bn_live 段) |
| 5 | `load_base` 双模式：官方骨干逐键 copy vs 全模型 strict=False | `ckpt_utils.py::load_base` |
| 6-7 | `ema_groups`：热身曲线（何时升满 0.999）；权重停住后 shadow 贴住实验 | `trainer.py::ema_update` |
| 8 | buffer 分类：running_mean/var 滑动平均、num_batches_tracked 照抄 | 同上 |
| 9 | 四参数组：对着分组规则数参数；解释"两分组名不副实"的修复注释 | `trainer.py`(groups 段) |
| 10 | autotune：batch=GB÷3 夹 [8,16] 公式；换显存档位看联动 | `main.py::autotune` |
| 11 | `gpu_metrics`：每步同步 vs 100 步一次同步的 stall 对比 | `pop_metrics` |
| 12 | batch IoU 的 GPU 化写法（union=0 clamp、只计含缺陷样本） | `trainer.py::train` |
| 13 | 暖机实验：首前向 vs 暖机后；解释"单图限时 10s"为什么需要暖机 | `main.py`（暖机段） |
| 14-15 | `ddp_spawn`：双进程真实跑通；改 world_size=2 损失曲线对齐；广播/all_reduce/参数一致三个断言 | `main.py::run_training/_worker` |

## 通关标准（全部达成即完成阶段 10）

- [ ] 解释 Conv-LoRA 筛选条件（1×1 且 in/out ≥64）与"键名不变，官方权重原样载入"
- [ ] 讲清 deepcopy 陷阱机理：闭包函数按引用共享 → EMA 拷贝执行训练侧原模块 →
      权重不生效 + SyncBN 单 rank all_gather 死锁；partial 经 memo 重绑的差别
- [ ] 写出 EMA 热身公式并算出升满 0.999 的步数（~9000）
- [ ] 解释 buffer 为什么要按浮点/整型分开处理
- [ ] 写出 autotune 三公式与四参数组的 lr 档位（lora/bb=1e-4，neck/head=主 lr）
- [ ] 说清 pop_metrics 为什么只同步一次；SyncBN 为什么只支持 GPU
- [ ] `pytest tests/` 全绿（含双进程 DDP 用例）

## 与主仓库的参数级对应

| 本工程 | 主仓库 |
|--------|--------|
| 筛选 `kernel==(1,1) & in/out≥64`、kaiming_uniform(a=√5)、B 零 | `seg.py::_attach_lora` |
| `partial(_lora_fwd, cl, scale)` 挂前向 | 同名（121~128 行注释全文对应） |
| `train()` 覆盖强制冻结段 eval + `_bn_live==0` 断言 | `OfficialEfficientViTSeg.train` / `trainer.py` |
| encoder.* 逐键 copy 白名单 / strict=False | `ckpt_utils.load_base` |
| `d=min(0.999,(1+step)/(10+step))`、`_foreach_mul_/_add_` | `trainer.py::ema_update` |
| 浮点 buffer 滑动 / 整数 buffer 照抄 / 只同步 requires_grad | `__init__`(_ema_* 段) |
| lora/bb 1e-4、neck/head 主 lr、`fused=True` | `trainer.py`(groups 段) |
| batch=GB÷3 夹 [8,16]、lr=4e-4×batch/8、workers=核÷2 夹 [2,8] | `main.py::autotune` |
| 指标 device 累加 + pop 一次 tolist | `pop_metrics` |
| 暖机 b16/b8 两规格 | `main.py` 暖机段 |
| gloo(CPU)↔NCCL(GPU)、env://、mp.spawn、张量广播 | `main.py::run_training/_worker` |

## 常见坑

- LoRA 挂 forward 用闭包：单卡测试全过，一开 EMA/多卡就"权重不生效/死锁"——
  最隐蔽的一类 bug，测试里用 deepcopy 改权重对照实验钉死；
- EMA 不做热身：前几十步 shadow 拖着随机初始化把评估指标拖到谷底；
- BN buffer 照抄参数的 EMA：整型 buffer（num_batches_tracked）做平均会变成小数；
- 构造后不调 .train()：冻结段 BN 一直处于构造时默认的训练态，running stats 被偷改
  （覆盖写了没人调用等于没写——体检断言必须显式触发）；
- 每步 `.item()` 取指标：4 个指标 4 次同步，GPU 流水线被打穿；
- CPU 上 convert_sync_batchnorm：直接 ValueError（SyncBN 仅支持 GPU 模块）。

## 下一步

进入阶段 11（指纹增量缓存与监控渲染）：`../stage11_cache_monitor/`。
