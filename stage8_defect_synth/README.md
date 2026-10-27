# Stage 8：LoRA 流匹配缺陷生成学习工程（2~3 周）

为读懂 `supervised_apps/train/src/train/lora_synth.py`（FLUX.1-Fill LoRA 训练 + 合成
缺陷图池生成，全工程 1671 行的"缺陷裂变工厂"）而设。前置：完成 `../stage2_pytorch/`
（训练循环）；建议先过 `train_mini/paste.py`（泊松贴图基础款）——本阶段是生成式路线。

本工程把真工程按"公式不变、载体缩小"的原则迷你化：FLUX.1-Fill 12 亿参数 → 3 层 MLP /
64 维玩具注意力；VAE 潜空间 → 2D 玩具分布；GPU 多卡 → 纯 CPU。**所有公式与阈值
逐项对应**，跑完就知道真工程每一步在做什么。

## 工程结构

```
stage8_defect_synth/
├── stage8/
│   ├── flow_matching.py     流匹配训练全公式 (t=sigmoid(u)/zt/target_v/token 加权 MSE)
│   ├── lora_linear.py       手写 LoRA 注入 nn.Linear (冻结+A/B 初始化+合并+瘦身 state_dict)
│   ├── soft_weight_crop.py  三区软权重 1.0→0.4 环带 / max+avg 下采样 / 预编码缓存 / 裁块
│   └── diff_extract.py      弹性形变模板 / 余弦羽化 / 差分滞后阈值提纯 / 质检门槛
├── tests/                   pytest 用例
├── conftest.py / requirements.txt（torch + numpy + opencv-python + pytest）
└── README.md
```

## 环境与运行

```bash
cd D:/ytc_robot/vision_learn/stage8_defect_synth
python -m venv .venv                      # 或复用 stage2 的 venv（同依赖）
.venv/Scripts/python.exe -m pip install -r requirements.txt

.venv/Scripts/python.exe -m stage8.flow_matching
.venv/Scripts/python.exe -m stage8.lora_linear
.venv/Scripts/python.exe -m stage8.soft_weight_crop
.venv/Scripts/python.exe -m stage8.diff_extract

.venv/Scripts/python.exe -m pytest tests/ -v
```

## 15 天学习日程（每天 ~1.5 小时）

| 天 | 内容 | 对应主仓库 |
|----|------|-----------|
| 1-2 | `flow_matching`：zt=(1−t)z0+t·noise 与 v=noise−z0 手推一遍；跑通采样看双峰 | `_lora_step2_worker`（644~667 行） |
| 3 | t=sigmoid(u) 为什么不是均匀采样（中间时段信息量最大，端点退化） | 同上 |
| 4 | token 加权 MSE：把某 token 权重置 0 看它是否"免死" | `token_mse·w_tok` 段 |
| 5-6 | `lora_linear`：恒等起步/冻结/合并三大性质测试；改 r 看 trainable 比例 | `LoraConfig(r=32,α=32)` 段 |
| 7 | PEFT target_modules 名单为什么是 to_k/q/v/out.0+add_*（注意力投影族） | 同上 |
| 8 | `soft_weight_crop`：三区权重数值剖面；改 dilate_px/edge_weight_min 看环带 | `_build_three_zone_masks` |
| 9 | max+avg 各半下采样：只用 avg 看小缺陷核峰值掉多少 | `_preencode_dataset`（w_pool 段） |
| 10 | 预编码缓存实验：二次访问零编码；对照真工程"text_encoder 用后即卸载" | 同上（pipe.text_encoder=None 段） |
| 11 | 裁块策略 k=ceil(60/N)≤4 与 >240 等距抽 | `_sample_crop_window` / `run_step1` |
| 12 | `diff_extract`：弹性模板 amp/σ 公式；对数均匀面积采样 | `deform_and_scale_template` |
| 13 | 余弦羽化与硬贴的边缘差；区外严格原样断言 | `cosine_feather_blend` |
| 14 | 差分滞后阈值：σ 的 MAD 估计、3.5σ 种子/1.8σ 生长、大缺陷兜底 | `compute_diff_and_sigma` / `extract_defect_mask_hysteresis` |
| 15 | 质检三门槛与 3 轮 1.15/1.25 补齐；合书复述"一条真实缺陷→合成图池"全链路 | Step4 门槛段 |

## 通关标准（全部达成即完成阶段 8）

- [ ] 默写流匹配四行公式（u/t/zt/v）并解释为什么目标是速度而不是噪声
- [ ] 解释 LoRA 的 B 零初始化为什么保证起步恒等、merge 后为什么逐位等价
- [ ] 写出三区软权重的数值剖面（核 1.0 → 8px 线性衰减到 0.4 → 背景 0）
- [ ] 解释 token 下采样为什么 max+avg 各半（峰值保留 vs 平滑过渡）
- [ ] 解释预编码缓存对显存的意义（训练循环里 VAE/text_encoder 都不在显存）
- [ ] 说清滞后阈值 3.5σ/1.8σ 各自的角色与大缺陷兜底的触发条件（核心占区 ≥70%）
- [ ] 列出质检三门槛与 3 轮差额补齐机制
- [ ] `pytest tests/` 全绿；流匹配采样中位距离 < 0.9；提纯 IoU > 0.6

## 与主仓库的参数级对应

| 本工程 | 主仓库 |
|--------|--------|
| `t = sigmoid(u).clamp(1e-3, 1-1e-3)` | `_lora_step2_worker` 同款 |
| `zt=(1−t)z0+t·noise`、`target_v=noise−z0` | 同名三行 |
| `loss=(token_mse·w).sum()/(w.sum()+1e-6)` | 同款 token 级加权 |
| `r=32, α=32, A=kaiming, B=0, target=to_k/q/v/out.0+add_*` | `LoraConfig` + 本课 targets 元组 |
| merge `W+=α/r·B·A` / 只导出 lora_* | `fuse_lora` / `get_peft_model_state_dict` |
| 核 1.0 / 8px 环带→0.4 / 标注 0 / 背景 0 | `_build_three_zone_masks` |
| `0.5·max_pool + 0.5·avg_pool` 下采样 | `_preencode_dataset` |
| 编码落盘 + 二次零编码 | latents/encoded.pt 缓存 |
| `k=ceil(60/N)≤4`、`MAX_LORA_CROPS=240` 等距抽 | `run_step1_for_zone_defect` |
| `amp=clip(短轴×0.15,1.2,5.5)`、`σ=clip(长轴×0.20,4,18)`、`s=clip(√(目标/当前),0.5,2)` | `deform_and_scale_template` |
| `α=0.5−0.5cos(πt)`、feather 4px | `cosine_feather_blend` |
| `δ=0.75·\|Δgray\|+0.25·\|Δgrad\|`、`σ=1.4826·MAD`、`clip(2,12)` | `compute_diff_and_sigma` |
| 3.5σ 种子 / 1.8σ 生长 / 7px 核容差 / 70% 兜底 | `extract_defect_mask_hysteresis` |
| 面积 ∈ [max(6,0.15core), 2.5core]、snr≥3.5、3 轮 1.15/1.25 | Step4 + `MAX_REFILL_ROUNDS` |

## 常见坑

- LoRA 的 A、B 都零初始化：梯度对 B 为 0、对 A 也为 0——死路；必须 A 随机 B 零；
- 流匹配采样方向搞反（沿 +v 积分）：噪声积分成更噪，采样图一片雪花；
- t 均匀采样而不用 sigmoid(u)：端点 (t≈0/1) 过采样，中间时段欠训练，采样中段漂移；
- 权重下采样只用 avg：小缺陷核（<16px）在 token 里被稀释成 0.2，等于不学；
- 滞后阈值只用一个阈值：高阈值漏边缘、低阈值满天星；两个阈值+连通域生长才两全；
- 弹性形变忘了位移场除以 std 再乘 amp：模糊后场强衰减到 ≈0，等于没形变。

## 下一步

进入阶段 9（零光晕泊松贴图与数据集工程）：`../stage9_paste_dataset/`。
