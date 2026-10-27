# Stage 6：精对齐与有效检测区学习工程（2~3 周）

为读懂 `supervised_apps/train/src/train/align.py`（逆向对齐 + ±10px 二次精对齐 + 真实
缺陷实例提取，全工程几何口径的心脏）而设。前置：完成 `../stage1_opencv/`（配准基础、
轮廓与形态学）；`train_mini/align.py`（ORB 对齐迷你版）先跑一遍更佳——本阶段是它的
"工业完全体"。

## 工程结构

```
stage6_precise_align/
├── stage6/
│   ├── corner_grid.py        分网格角点采样 / 粗尺度 8-DOF 单应 / 位姿门控 / 失败隔离
│   ├── normal_snap.py        ±10px 法向剖面吸附（训练模式）——compute_effective_zone 上半
│   ├── smooth_bridge.py      曲率保持去锯齿 + 大窗架桥（推理模式）——下半
│   └── defect_instance.py    最大交集归属 / 环带对比度 / 去背景导出 / 规格解算 / 核心提纯
├── tests/                    pytest 用例
├── conftest.py / requirements.txt（numpy + opencv-python + pytest）
└── README.md
```

## 环境与运行

```bash
cd D:/ytc_robot/vision_learn/stage6_precise_align
python -m venv .venv                      # 或复用 stage1/stage2 的 venv（同依赖）
.venv/Scripts/python.exe -m pip install -r requirements.txt

.venv/Scripts/python.exe -m stage6.corner_grid
.venv/Scripts/python.exe -m stage6.normal_snap
.venv/Scripts/python.exe -m stage6.smooth_bridge
.venv/Scripts/python.exe -m stage6.defect_instance

.venv/Scripts/python.exe -m pytest tests/ -v
```

## 15 天学习日程（每天 ~1.5 小时）

| 天 | 内容 | 对应主仓库 |
|----|------|-----------|
| 1-2 | `corner_grid`：跑通纹理图对齐，看四角复现 < 2px；改 H_gt 平移量看门控何时拒绝 | `align.py::Point8DofAligner.align` |
| 3 | `spatial_select`：把 6×6 网格改成 3×3 / 12×12 看内点与稳定性变化，理解空间均匀的意义 | `align.py::spatial_select` |
| 4 | 粗尺度还原：直接全分辨率求 Hs 与 S⁻¹·Hs·S 对比（耗时/精度）；读 `swap_rc_xy` 理解 Halcon (row,col) 陷阱 | `_to_struct_scale` / `project_mask_to_query` |
| 5 | 位姿门控三条件逐个破坏（大位移 / 离谱尺度 / 内点不足），看失败字典的 reason | `align.py::align`（安检段） |
| 6 | 失败隔离：制造失败项跑 `quarantine_failed`，读 moved.jsonl 审计行 | `align.py::align_all`（failed_items 段） |
| 7-8 | `normal_snap`：跑锯齿分区吸附，看 IoU 与区内干净度；改 SHOULDER_IN=0/3.0 对比区内越区像素 | `compute_effective_zone`（训练分支） |
| 9 | 剖面采样：改 snap_px=5/20；把轮廓点数抬过 32767 验证分块 remap 的存在意义 | `_remap_1d_profiles` |
| 10 | 方向定向实验：注释掉 `orient_normals` 看吸附去哪了（对面暗缝） | 同上（dir_sign 段） |
| 11-12 | `smooth_bridge`：缺口覆盖对比训练/推理两种分区；`bridge_shifts` 单元里看"只拉凹陷不推正常段" | `compute_effective_zone`（推理分支） |
| 13 | 曲率保持：`smooth_keep_curvature` vs 直接大核高斯平滑圆轮廓，量面积收缩 | 同上（win_g 段） |
| 14 | `defect_instance`：归属/环带对比度/导出/规格全跑一遍；对照 real_*.png 的生成逻辑 | `_extract_product_defect_instances` / `auto_bind_zones` |
| 15 | 合书重写 `snap_zone_train`（不看源码），测试全绿即毕业 | — |

## 通关标准（全部达成即完成阶段 6）

- [ ] 说清"图不动、蒙版动"的逆向对齐哲学与三类安检（内点数 / 四角位移 / 尺度行列式）各自拦什么病态解
- [ ] 解释粗尺度求解后为什么必须 S⁻¹·Hs·S 还原，以及 H /= H[2,2] 的齐次归一化意义
- [ ] 写出法向剖面吸附的 5 步（切向→法向 → remap 剖面 → 方向定向 → 加权峰值 → 肩部+中值平滑）
- [ ] 解释训练模式收 1.5px / 推理模式收 1.0px + 架桥的差异动机（区内干净 vs 边缘缺肉包进区）
- [ ] 解释曲率保持 = 平滑主形 + 平滑残差，为什么直接平滑整条轮廓会缩圆弧
- [ ] 说清最大交集归属为什么用"圆滑区"而不用"吸附区"、以及 < 4px 回退投射区的兜底
- [ ] `pytest tests/` 全绿；`corner_grid` 四角复现 < 2px；推理分区在缺口处边界不下凹（架桥拉平）

## 与主仓库的参数级对应

| 本工程 | 主仓库 |
|--------|--------|
| `GRID=6 / PER_CELL=30` | `align.py::GRID / PER_CELL` |
| `COARSE_W=512`、`H = inv(S)@Hs@S; H/=H[2,2]` | `Point8DofAligner(coarse_width=512)` |
| `MIN_INLIERS=50` | `align()`：内点 < 50 失败 |
| 四角位移 ≤ 0.15·max(H,W)、det∈(0.5, 2.0) | `align()` 位姿门控常量 |
| `swap_rc_xy`（置换矩阵夹乘） | `project_mask_to_query` |
| `SNAP_PX=10 / PEAK_GRAD_MIN=4.0 / SHOULDER_IN=1.5` | `SNAP_PX` 与训练分支 |
| 梯度衰减 `1 − 0.015·\|t\|`、滑窗中值 pad=min(5, n//6) | 同名公式 |
| 肩部 1.0px + 55 百分位架桥 + 高斯核 σ=pad/2.5 | 推理分支 |
| `EXPORT_PAD_PX=8`、环带 dilate15−dilate5、p80≥6 | `_extract_product_defect_instances` |
| 面积 lo=max(18, p2×0.75) / hi=max(lo×1.5, p98×1.30)、对比度 [16,195] | `auto_bind_zones`（zd_specs 段） |
| 核心提纯 max(6, 0.28·p85) + 3×3 膨胀回 | `assets.py::refine_asset_patch_and_mask` |
| `quarantine_failed` + moved.jsonl | `align_all` 失败隔离段 |

## 常见坑

- 法向没定向就找峰值：亮零件吸到对面暗背景的"反向梯度沿"，边界整体外弹 10px；
- remap 的 map 超过 32767 点：OpenCV SHRT_MAX 断言直接崩，必须分块（16000）拼接；
- 层级轮廓直接全部 FILLED：孔被最后一个画上去的外环盖掉——必须按面积从大到小、父环 255 子环 0；
- 环带对比度用均值而不是 p80：一颗椒盐噪点就把背景中值带偏；用中值背景 + 80 分位缺陷差；
- 齐次矩阵忘了 `/= H[2,2]`：cv2 内部会归一化但自己读 H 元素对比真值时全错位。

## 下一步

进入阶段 7（素材配额预算数学）：`../stage7_quota_budget/`。
