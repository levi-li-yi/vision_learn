# Stage 7：素材配额预算数学学习工程（1~2 周）

为读懂 `supervised_apps/train/src/train/budget.py`（训练底图分卷、图块配额、合成张数
同源导出的统一预算解算器）而设。前置：无深度依赖，纯数学 + NumPy；建议先过
`stage0_fundamentals`。这是全工程"每天要生成多少张合成缺陷图、每轮贴多少块"的
财政部门——所有配额都从同一套数学模型里长出来，改一处全局联动。

## 工程结构

```
stage7_quota_budget/
├── stage7/
│   ├── split_seed.py          8:2 确定性分卷 / 逐点位 md5 种子 / 零干净块置换
│   ├── waterfill.py           带上下界约束的注水分配 W=√(T·D)、L 保底、U=5T 封顶
│   ├── largest_remainder.py   最大余数法取整（总和严格守恒 + 种子化平票扰动）
│   ├── grid_patch.py          0.75 步长网格切块 / 脏块剔除 / 骑缝小检测区居中补块
│   └── quota_model.py         B_epoch/num_shards 全模型 + 合成配额同源导出 E/(10D)
├── tests/                     pytest 用例
├── conftest.py / requirements.txt（numpy + opencv-python + pytest）
└── README.md
```

## 环境与运行

```bash
cd D:/ytc_robot/vision_learn/stage7_quota_budget
python -m venv .venv                      # 或复用 stage1/stage2 的 venv（同依赖）
.venv/Scripts/python.exe -m pip install -r requirements.txt

.venv/Scripts/python.exe -m stage7.split_seed
.venv/Scripts/python.exe -m stage7.waterfill
.venv/Scripts/python.exe -m stage7.largest_remainder
.venv/Scripts/python.exe -m stage7.grid_patch
.venv/Scripts/python.exe -m stage7.quota_model

.venv/Scripts/python.exe -m pytest tests/ -v
```

## 10 天学习日程（每天 ~1.5 小时）

| 天 | 内容 | 对应主仓库 |
|----|------|-----------|
| 1 | `split_seed`：分卷比例/上限/确定性；加点位验证"别的点位不动我的牌" | `split_train_earlystop` |
| 2 | 零干净块置换：小检测区的救命机制；全零时 SystemExit 的哲学（大声失败） | `_preslice_clean_tile_counts` |
| 3-4 | `waterfill`：跑 4 档预算看分配表；手算一遍 √(T·D) 权重；对抗性边界观察"碎屑"现象 | `_waterfill_allocate` |
| 5 | `largest_remainder`：对比朴素 round 的总和漂移；平票扰动换 seed 看分配变化 | `largest_remainder_round` |
| 6 | `grid_patch`：0.75 步长、并集 16px / 单区 8px 门槛、标注膨胀 4px 剔脏块 | `build_image_grid_for_zones` |
| 7 | 骑缝补块实验：把小区挪到网格缝上看"完整版"补块出现 | 同上（补块段） |
| 8-9 | `quota_model`：改 M_total 看 v_base 从 5→1；改 D 看 E 变化与合成配额联动 | `compute_product_budget` |
| 10 | 合书默写"一条标注如何决定明天要合成多少张图"全链路，测试全绿即毕业 | — |

## 通关标准（全部达成即完成阶段 7）

- [ ] 写出 v_base / B_epoch / L / U / W 五个公式并解释各自防的病（过拟合 / 小类别饿死 / 大区垄断）
- [ ] 解释逐点位独立种子为什么是缓存指纹体系的前提（跨次运行稳定性）
- [ ] 说清注水分配的收敛过程（钉下界→钉上界→自由按权重）与"永不超发"的边界条件
- [ ] 解释最大余数法 + 平票扰动相对朴素 round 的两个收益（总和守恒 / 不偏袒排序）
- [ ] 说清骑缝补块解决什么问题（小检测区被网格线切碎 → 至少一块完整版）
- [ ] 从 E_{p,z} 一路除出合成张数：÷D（均分缺陷）÷2（一半用真实）÷5（增强复用）
- [ ] `pytest tests/` 全绿

## 与主仓库的参数级对应

| 本工程 | 主仓库 |
|--------|--------|
| `MAX_EARLYSTOP_PER_POINT=10`、20% 分卷 | budget.py 同名常量 |
| `point_seed_offset`：md5 前 4 字节 % 1000003 | `_point_seed_offset` |
| W=√(T·D)、L=min(5T, max(100D, T/epochs))、U=5T | `compute_product_budget` |
| `TARGET_PASSES_PER_POINT=200 / MAX_VARIANTS=5 / MAX_EPOCH_POS_TILES=16000` | 同名常量 |
| ΣL > 预算时按比例压低下界（每区保 1） | `compute_product_budget`（scale_low 段） |
| `num_shards = max ceil(T/E)`（≤epochs，保证 100% 底图参训） | 同名公式 |
| `SYNTH_AUG_REUSE_FACTOR=5`、q_raw=E/(10D)、钳位 [60, 600] | 同名常量与合成导出段 |
| `ANN_SAFE_PAD_PX=4 / MIN_IN_ZONE_PX=8 / MIN_UNION_ZONE_PX=16` | 同名常量（与 dataset.py 严格一致） |
| `largest_remainder_round(seed+37)` | 跨点位合成配额分回 |

## 常见坑

- 分卷用全局种子：新增一个点位后所有点位重新洗牌，缓存指纹全废——必须逐点位偏移；
- 注水忘了迭代钉死：一轮就把越界者剔除后不再重算，剩余预算按旧权重分，总量对不上；
- 朴素四舍五入分配额：每天差 1~2 张，增量补齐循环永远跑不完；
- 网格边块不钳位去重：`min(i·stride, W−ts)` 会出现两块同坐标或越界坐标；
- 补块忘了剔脏：骑缝补的块恰好压在标注上，把缺陷贴进了"干净底图"。

## 下一步

进入阶段 8（LoRA 流匹配缺陷生成）：`../stage8_defect_synth/`。
