# Stage 4：无监督异常检测学习工程（3~5 周）

为读懂 `vision-ai-training/unsupervised_training/` 全部而设。前置：完成
`../stage2_pytorch/`（训练循环）与 `../stage3_segmentation/`（分割与阈值扫描）。

## 工程结构

```
stage4_anomaly/
├── stage4/
│   ├── paradigm_map.py            四范式一文件：AE/师生/记忆库/CutPaste(特征版)
│   ├── pdn_and_loss.py            PDN 双胞胎(conv1 继承/冻结) + 0.1% 量化困难损失
│   ├── calibration_and_auc.py     分位标定/退化阶梯/桶门槛/AUC/TopK100——core/utils.py 复刻
│   └── demo_l_teacher_student.py  ★Demo L：手写 T-S 端到端(合成 MVTec 替身)
├── tests/                         pytest 用例
├── conftest.py / requirements.txt（torch + numpy + pytest）
└── README.md
```

## 环境与运行

```bash
cd D:/ytc_robot/vision_learn/stage4_anomaly
python -m venv .venv                      # 或复用 stage2 的 venv（同依赖）
.venv/Scripts/python.exe -m pip install -r requirements.txt

.venv/Scripts/python.exe -m stage4.paradigm_map
.venv/Scripts/python.exe -m stage4.pdn_and_loss
.venv/Scripts/python.exe -m stage4.calibration_and_auc
.venv/Scripts/python.exe -m stage4.demo_l_teacher_student --save heat.png

.venv/Scripts/python.exe -m pytest tests/ -v
```

## 25 天学习日程（每天 ~1.5 小时）

| 天 | 内容 | 对应主仓库 |
|----|------|-----------|
| 1-2 | `paradigm_map`：四范式各跑一遍，看 AUC；殊途同归"只学好件" | 全局观 |
| 3-4 | 论文 CutPaste(CVPR21)：合成异常自监督——本工程 test 考卷源头 | 无监督 3.1 合成线 |
| 5-6 | 论文 DRAEM(ICCV21)：合成+判别双网；对比本工程"仅 test 用合成" | 同上(刻意不做训练期合成) |
| 7-9 | **作业 Demo I**：装 `anomalib`，MVTec AD 跑 PatchCore + EfficientAD，看 anomaly map 与 AUC | — |
| 10-11 | `pdn_and_loss`：conv1 继承/冻结三动作；量化困难损失为什么只罚 0.1% | `modeling.py` + `trainer.py:118-124` |
| 12-13 | 论文 EfficientAD(CVPR24)：PDN 直系源头；**作业 Demo J**：读 `nelson1425/EfficientAD` 源码，逐行对比本工程 `modeling.py` 改了什么 | `core/modeling.py` |
| 14-16 | `calibration_and_auc`：90/99.5 分位定尺、ungated→gated→eps 阶梯、桶门槛=4、TopK100 抗噪 | `core/utils.py`（含金量最高，逐行） |
| 17-18 | ★`demo_l_teacher_student`：亲手跑通 + 读代码；改 steps/贴块大小看 AUC 变化 | 缩影版全线 |
| 19-21 | **作业 Demo L-MVTec 版**：把 demo_l 的 make_ok 换成 MVTec bottle 类别（`--data` 自行接线），真图真 AUC | — |
| 22-23 | 回读 `train_entry.py`：三池划分/真实 NG 预切/AUC 门控保存 | 无监督 2.2/6.1 |
| 24-25 | 合书重写 demo_l（不看源码），测试全绿即毕业 | — |

## 通关标准（全部达成即完成阶段 4）

- [ ] 说清四范式各自打分函数，以及本工程为何选师生范式（像素级距离图 + 可分桶标定）
- [ ] 解释"student 模仿不上 teacher 的地方就是异常"的机理（teacher 见过世面/学生只学好件）
- [ ] 解释"为什么量尺子必须用 val、不能用 train"（背题 -> 距离虚低 -> 尺子量小 -> 冤案）
- [ ] 解释"为什么每个拍摄位置(桶)要独立标定 mean/std/q/阈值"（背景桶 vs 纹理桶正常波动天差地别）
- [ ] 解释"打分取 TopK100 像素均值而不是最大值"（单噪点劫持）——测试里有定量断言
- [ ] 写出退化阶梯三级及各自后果（eps 档分数不可跨模型比较）
- [ ] `pytest tests/` 全绿；Demo L 端到端 AUC > 0.85（合成）/ > 0.7（MVTec bottle 作业）

## 与主仓库的参数级对应

| 本工程 | 主仓库 |
|--------|--------|
| `quantile_hard_mining_loss(q=0.999)` | trainer.py 前 0.1% 困难挖掘 |
| `map_quantiles(0.9, 0.995)` | `core/utils._map_quantiles` |
| `robust_quantiles` 三级返回 | `_robust_quantiles` ungated/gated/eps |
| `Q_SCALE_MIN_REL=1e-6` | 同名常量 |
| `MIN_BUCKET_SAMPLES=4` | 同名常量 |
| `bucket_key(x,y)=f"x{x}_y{y}"` | 四方共用桶键 |
| `topk_mean(k=100)` | test() TopK100 打分 |
| `build_pair` conv1 继承 | `build_teacher_student_models` |
| Demo L 的 `train_student` | 教材纯好件 + teacher 归一化后抄写 |

## 常见坑

- kthvalue 的 k 必须**钳位到 [1,n]**：小样本时 `int(n*0.995)` 可能为 0 -> 直接崩；
- 分位"退化"不报错只返回同一数——不主动判 `q_end>q_start` 就会除零出 inf；
- teacher 不冻结(`requires_grad=False`)会让"抄作业"变成"两人一起跑"，损失照样降但异常图全黑；
- 训练 student 时忘了 teacher 归一化：384 通道量纲不一致，距离被大通道垄断，异常图全是噪点。

## 下一步

进入阶段 5（回到本仓库总装，2~4 周）：`../stage5_repo_integration/`。
