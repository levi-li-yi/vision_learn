# Stage 9：零光晕泊松贴图与数据集工程学习工程（2~3 周）

为读懂 `supervised_apps/train/src/train/dataset.py`（873 行的在线贴图数据集：正负 1:1、
一块多区、区内/区边缘五五分、零光晕泄漏泊松融合）而设。前置：完成 `../stage1_opencv/`
（blending 基础）与 `train_mini/paste.py`（泊松基础款）；本阶段是工业完全体。

## 工程结构

```
stage9_paste_dataset/
├── stage9/
│   ├── paste_zero_halo.py    零光晕泄漏泊松七件套（半压缩/eval_mask/α环/极性/重标定/门控/carve）
│   ├── place_rules.py        区内/区边缘五五分 + 可见性复检 + 上下文镜像窗口
│   ├── deck_sampling.py      牌靴无放回抽样 / 逐样本确定性 rng / EpochShardSampler
│   └── tile_cache.py         LRU 预算缓存 / 一块多区互斥贴图 / 切块抖动
├── tests/                    pytest 用例
├── conftest.py / requirements.txt（numpy + opencv-python + torch + pytest）
└── README.md
```

## 环境与运行

```bash
cd D:/ytc_robot/vision_learn/stage9_paste_dataset
python -m venv .venv                      # 或复用 stage2 的 venv（同依赖）
.venv/Scripts/python.exe -m pip install -r requirements.txt

.venv/Scripts/python.exe -m stage9.paste_zero_halo
.venv/Scripts/python.exe -m stage9.place_rules
.venv/Scripts/python.exe -m stage9.deck_sampling
.venv/Scripts/python.exe -m stage9.tile_cache

.venv/Scripts/python.exe -m pytest tests/ -v
```

## 15 天学习日程（每天 ~1.5 小时）

| 天 | 内容 | 对应主仓库 |
|----|------|-----------|
| 1 | `paste_zero_halo` 朴素克隆对照：跑出光晕像素与"平坦缺陷被解成背景"两个现场 | `paste_poisson` 动机 |
| 2-3 | 七件套逐个关掉跑一遍（eval_mask/α 环/极性/重标定/门控），看各自坏了什么 | `paste_poisson` 172~335 行 |
| 4 | 区外 carve/built：贴边缺陷跨区过渡与"label=0 一个不动"的共存 | 同上（末段） |
| 5-6 | `place_rules`：五五分统计、区内 0.75/边缘 0.15 门槛、截断形态分布 | `try_place` |
| 7 | 可见性复检：Δ=3 的"隐形缺陷"为什么必须拒收（标签 1 却看不见 = 教模型瞎） | `VISIBLE_MIN_DIFF` 段 |
| 8 | 上下文镜像窗口：margin 从哪来（max(patch)//2+DILATE_MAX+4），为什么要 REFLECT_101 | `context_window` / `_tile_context` |
| 9-10 | `deck_sampling`：牌靴覆盖率/确定性实验；换 deck 边界看"换牌"时机 | `assets._get_deck_perm` |
| 11 | 逐样本 rng 公式背诵与回放实验（同 (样本,轮) 全卡同增强） | `dataset.__getitem__` |
| 12 | EpochShardSampler：轮号编码反解、2 分片合覆盖、DDP 条带切分 | `EpochShardSampler` |
| 13 | `tile_cache`：LRU 预算淘汰顺序；cache_mb 公式手算 | `_orig_gray` / `main._worker` |
| 14 | 一块多区互斥贴图 + 切块抖动的免费增强效应 | `__getitem__`（occupied/jitter 段） |
| 15 | 合书画出"一个 batch 的旅程"（索引→反解→抖动→贴图→归一化），测试全绿即毕业 | — |

## 通关标准（全部达成即完成阶段 9）

- [ ] 说清朴素 seamlessClone 的两大罪：掩码外光晕泄漏 + 平坦区域解成背景（梯度 0 → 解=dst）
- [ ] 默写零光晕七件套并解释 label=0 恒不动的实现机制（eval_mask 内才给权重）
- [ ] 解释五五分放置动机（只学完整缺陷会漏检边缘缺陷）与两种模式的区内占比门槛
- [ ] 写出逐样本 rng 公式，解释"跨卡同增强"对可复现的意义
- [ ] 解释牌靴相对"每次随机抽"的两个保证（全覆盖周期 / 跨进程一致）
- [ ] 解释轮号编码如何让 persistent_workers 感知分片轮换（不发广播）
- [ ] 写出 cache_mb 解算公式与持有者数的含义
- [ ] `pytest tests/` 全绿；零光晕泄漏 = 0 像素断言通过

## 与主仓库的参数级对应

| 本工程 | 主仓库 |
|--------|--------|
| `DILATE_MIN/MAX=5,7`（rng.randint） | `paste_poisson` 开头 |
| 半压缩 `(I−med)×0.5+128`、残差 `2.0×Δ`、`res=0.5δ+0.5src_d` | 同名公式 |
| `eval_mask=(m>0.30)∧zone`、`m>0.35` 求解域 | 同名两阈值 |
| `α=0.55+0.45(dt−1)`、外扩域 dilate、`hole_gate=(I−20)/35` | 同名行 |
| 极性 ±1 翻转 + ±4 截断、重标定 clip(0.5,6)·target | 同名段 |
| `BOUNDARY_PLACE_PROB=0.5`、边缘带 d≤2.5、深处 d≥min(d_max/2, 0.75r_eq) | `try_place` |
| 门槛 0.15/0.75、`VISIBLE_MIN_DIFF=8`（均值 8 或 p75 12） | `_paste_one_zone` |
| `context_window` REFLECT_101 | 同名函数 |
| 牌靴 md5 五元组 + `deck_id=g//N` | `assets._get_deck_perm` |
| `rng=Random(seed·1000003+epoch·100003+variant·987654+pool_idx)` | `__getitem__` |
| `index=epoch·len(pool)+idx`、正负成对、DDP drop_last | `EpochShardSampler` |
| cache_mb=(可用−8G)×0.6÷持有者、LRU move_to_end | `_orig_gray` / `main.py` |
| jitter ±stride/2 × 6 次、避标注、并集≥16 | `__getitem__` jitter 段 |

## 常见坑

- 拿 seamlessClone 输出当结果直接写回：平坦背景区解成 dst，缺陷内容凭空消失；
- eval_mask 只用 label 不交 zone：贴边缺陷把区外背景也改了，label=255 区出现"幽灵改动"；
- 极性不强制：泊松解个别像素翻极性，暗缺陷里冒出反色亮噪点；
- 逐样本 rng 用全局 random：多 worker 各自继承同一状态，八张图增强完全一样；
- LRU 预算不除持有者数：4 个 worker 各按全量预算缓存 → 内存 ×4 直接 OOM；
- 抖动后不查标注：把窗口挪到有标注的块上，"干净底图"里混进真缺陷被标成 0。

## 下一步

进入阶段 10（训练器进阶：Conv-LoRA / EMA / 多参数组 / DDP）：`../stage10_trainer_advanced/`。
