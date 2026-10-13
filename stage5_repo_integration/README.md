# Stage 5：回到本仓库总装（2~4 周）

前置：阶段 0~4 全部通关。本阶段没有新知识点——只有"把已学的每一格
零件装回 `vision-ai-training` 原厂"，以及四个从小到大的上手实验。

## 总装阅读顺序（每步带自问清单）

| 步 | 读什么 | 自问清单（全部能答才走下一步） |
|----|--------|------------------------------|
| 1 | 根 README + 两子 README | 两线粒度差异？数据拓扑谁共享？wandb 怎么配？ |
| 2 | `confs/` 四个配置文件 | 每个参数属于阶段几？(warmup→S2、focal_γ→S3、0.999→S4) |
| 3 | `supervised_training/data_utils/data.py` | 三道滤网？桶均衡算几次？warpAffine 搬的是谁？ |
| 4 | `core/seg.py` + `core/loss_utils.py` | 65,536 怎么来？总分公式哪行？ |
| 5 | `supervised_training/trainer.py` + `train_entry.py` | 五连击顺序？断点/部署分工？ |
| 6 | `unsupervised_training/` 全部 | 三池为何 test=全量？AUC 门控谁换谁？ |
| 7 | `eval.py` 双模式 + `tools/` | Micro 两把尺？qr 的 note 有哪几种？ |

配套文档（都在 `vision-ai-training/docs/`，本阶段全程开着对照）：
- `07-监督线原理文档.html`——为什么这么设计（时间线叙事）
- `08-无监督线原理文档.html`——姊妹篇
- `09-监督线代码解读文档.html`——每一行怎么写（15 文件全覆盖）

## 四个上手实验（从小到大；本机 RTX 3060 参数已适配）

### 实验 1：合成预览（无 GPU，10 分钟）

```bash
cd D:/ytc_robot/vision-ai-training/supervised_training
python tools/preview_synthesis.py --product_code <机种> --fixture_code <夹具> \
    --dataset_root <数据根>
```
任务：改 `defect_scale_range`(0.5,1.3) / `place_min_coverage` 0.6→1.0，
各重跑一次，**预告片里贴纸数量/形态怎么变**——把感受写进实验笔记。

### 实验 2：跑通一次 fresh 全流程（小任务）

```bash
python train_entry.py --task_whitelist "<机种>/<小夹具>" \
    --batch_size 1 --accumulation_steps 8 --epochs 3 \
    --ckpt_dir ./checkpoints --sample_dir ./sample
```
任务：读日志的每一段（资产审计/桶均衡/损失/落盘）；确认 meta.json、
{时间戳}.pth、regions/ 三个产物都出现了。

### 实验 3：消融——关掉 Dice

```bash
# 基线一次 + --dice_weight 0 一次（其余全同）
python train_entry.py --task_whitelist "..." --epochs <小> \
    --batch_size 1 --accumulation_steps 8 --ckpt_dir ./ck_baseline
python train_entry.py --task_whitelist "..." --epochs <小> --dice_weight 0 \
    --batch_size 1 --accumulation_steps 8 --ckpt_dir ./ck_nodice
python eval.py --ckpt_dir ./ck_baseline/<机种>/<夹具> --eval_data_dir <测试集>
python eval.py --ckpt_dir ./ck_nodice/<机种>/<夹具>   --eval_data_dir <测试集>
```
任务：对比两张阈值扫描表的 best-F1 与最优阈值位置——验证"Dice 是
压分主导项"的设计陈述（07 文档 5.1）。

### 实验 4：无监督一个点位 + 读标定日志

```bash
cd ../unsupervised_training
python train_entry.py --task_whitelist "<机种>/<夹具>/<点位>" --epochs 3
```
任务：把日志里的 `[CALIB]` 行逐条对应到 `core/utils.py` 的函数：
teacher_normalization（几桶达标）/ map_normalization（global=ungated?）/
bucket_calibration 入表几桶 / AUC 门控结果。

## 通关标准（全达成 = 整条学习线毕业）

- [ ] 独立回答此前讨论过的一切：为什么 Cut-Paste 在线合成？底纸为什么不是
      标件图？region 为什么参与裁答案？拉扯问题的机理与实测结论？
- [ ] 改任何一个 DataArgs 参数，能说出管线哪一步受影响（对照 09 文档速查表）
- [ ] 四个实验各有笔记与结论
- [ ] （进阶）给 07/08/09 三份文档各提一条勘误或补充建议——提得出来，
      说明你已经从读者变成了审查者

## 恭喜

走到这里，`vision-ai-training` 对你不再是黑箱：**数据怎么造、样本怎么生、
模型怎么学、阈值怎么定、产物怎么交付**——每一环你都在迷你工程里亲手
写过一遍，原厂只是把每一环放大并加上了产线的约束。
