# train_mini —— 迷你同构训练学习工程(≤2GB 显存跑通完整流程)

真训练工程(`supervised_apps/train`)的**全流程缩影**:事件线、考试选型、交付契约逐项一致,
组件换成轻量同型件——**学完它等于学完真架构**。产出可直接被姊妹工程 `infer_lite` 服务。

## 与真工程的映射表

| 事件线阶段 | 真工程 | 迷你同构(本工程) |
|---|---|---|
| 0 体检 | checkup.py(域分区/同分辨率/实例数≥4/增量缓存) | `zones.scan_point`(域分区/同分辨率/考卷隔离;实例数与缓存从简) |
| 1 对齐 | Halcon Point8DofAligner(8-DOF 单应,内点≥50)+ 有效检测区(±10px 吸附+圆滑) | `align.OrbAligner`(ORB+RANSAC 单应,内点≥30,宽512估计+S⁻¹HS 还原)+ 投影分区(**吸附/圆滑从简,差异点**) |
| 哲学 | 图不动,蒙版动(dev 版) | **完全一致**(`zones.effective_zones`:分区投到每张原图自己的坐标系;失败回退朴素分区) |
| 2 缺陷素材 | FLUX.1-Fill LoRA 生成 + 真实缺陷小图 | **泊松融合贴图**(旧方案回归,`paste.py`:外扩5-7px NORMAL_CLONE/对数均匀抽样/随机旋转)+ 真实缺陷小图(`patches.py`,外扩8px,掩码∩分区) |
| 3 切块配额 | budget.py(全图网格+骑缝补块+面积配额+增量缓存) | `budget.py`(全图网格 0.75 步长;**跳过含标注脏块**;补块/配额/缓存从简) |
| 4 训练 | EfficientViT-SAM L1+LoRA+级联头,Focal+Dice×3,bf16,EMA,DDP | **TinyUNet(~0.4M)**,同 Focal(α.4,γ2)+Dice×3、同早停(8轮后连4轮不降);LoRA/EMA/DDP 从简 |
| 5 考试交付 | exam.py(帐篷窗/交集/hit/quality_score/best+deploy) | `exam.py`(**逐公式逐字段同款**) |

**交付契约 100% 一致**:ckpt 键 `{"weight","step","epoch","intersection","hit_defects","quality_score"}`、
best 命名 `best_int{交集}_ep{轮}.pth`、manifest `{weight,tile_size,threshold:0.5,points.{pid}.{ref_name,json,ref_image,zones}}`、
deploy 目录结构——`infer_lite` 切到 `mini_model` 构造器即可直接服务。

## 显存预算(为什么 2GB 够,实测口径)

| 项 | 量 |
|---|---|
| TinyUNet 参数/梯度/AdamW(~0.4M×4×4B) | ~7MB |
| 激活(256×256,batch 4,AMP) | ~300-500MB |
| **训练峰值** | **<1GB**(调 batch 8 仍宽裕) |
| 考试整图推理(batch 8) | <800MB |
| infer_lite 服务(权重驻 CPU,动态分块) | <300MB |

## 使用

```
datasets/                         # 与真工程同布局(define_masks 产物直接可用)
└─ {产品}/{点位}/
    ├─ {基准图名}.json            # define_masks 分区(必含 domain)
    ├─ *.bmp (+ *.bmp.tmp)        # 训练池(OK+NG;NG 带 .tmp 标注)
    └─ val/*.bmp (+ .tmp)         # 独立考卷(100% 隔离)

python main.py --product HT12
# 产出 models/HT12/best_int*.pth + deploy/(v2 契约)
# → 整目录拷入 infer_lite 的 {SUPERVISED_CKPT_DIR}/HT12/ 即可 POST /detect
```

训练观测:控制台日志之外,另有零依赖监控页(真工程 8300 监控 web 的同构缩影)——
训练进程把事件线与逐轮指标原子写 `.tmp/{产品}/web/state.json`(runtime_state.py),
`web.py` 常驻读该文件渲染;两进程零耦合,先后启动都行,训练结束后页面仍可回看。

```
python web.py --product HT12          # 另开终端 → http://127.0.0.1:8301
# 逐轮 train/val loss 与考卷 quality_score 曲线 + 事件线 + best/早停状态,2s 自动刷新
# 内网观看加 --host 0.0.0.0;端口可 --port 或 config 的 web_port 改
```

依赖:`torch / opencv-python / numpy / Pillow`(无 halcon、无 FLUX、无 diffusers)。
配置见 `config/app_config.jsonc`(tile 256 / batch 4 / 早停参数与真工程同口径)。

## 刻意从简的差异清单(学习时心里有数)

1. 无 FLUX 生成 → 泊松贴图(旧方案回归):学习目标是"缺陷扩增要有边缘自然感",两条路都成立;
2. 无 ±10px 吸附/圆滑 → 投影分区直用:真工程"有效检测区"是精度增强,架构位与本工程相同;
3. 无增量缓存/骑缝补块/面积配额/EMA/DDP:事件线与口径不变,工程化增强从简;
   监控 web 为零依赖轻量版(state 文件 + 曲线/事件线渲染;无 NG 热力图、整图按需推理、训练子进程管理);
4. 对齐用 OpenCV(x,y 口径直连)替代 Halcon(row,col+交换):免 license,思想同构;
5. .tmp 标注解析/分区 json/RLE/网格/帐篷窗/0.5 阈值/quality_score 公式与真工程**逐字一致**。
