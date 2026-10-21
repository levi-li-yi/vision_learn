# infer_lite —— 推理学习工程(v2 副本 + 低显存配置)

`vision-ai-inference-v2` 的**原样副本**(纯监督 · 布局 B · 训练同款口径),叠加三层学习适配:

1. **`configs/low_vram.ps1`**——≤2GB 显存启动配置(`GPU_MEM_SAFETY=0.7 / BATCH_MAX=8 / AMP_DTYPE=fp16`);
2. **`mini_model.py`**——TinyUNet 构造器,可直接服务姊妹工程 `train_mini` 交付的 best 权重(免建 wheel);
3. 本 README 的三种玩法。

推理口径与真工程**完全一致**(灰度×3 / 全图网格 / 帐篷窗 / 有效检测区门控 / manifest 选权重 / (文件名,mtime) 热更新),区别只在参数档与模型体量——学它等于学 v2。

## 三种玩法

| 玩法 | 前置 | 启动 |
|---|---|---|
| A. 服务 train_mini 产物(推荐起步) | 装好 torch/cv2/fastapi 等(**无需** wheel、无需 halcon) | `$env:SUPERVISED_MODEL_MODULE="mini_model"; $env:SUPERVISED_MODEL_BUILDER="build_model"; $env:REGION_ALIGN="0"; . .\configs\low_vram.ps1; python app_run.py` |
| B. 服务真工程产物(EfficientViT-SAM) | 构建 vendor wheel + mvtec-halcon(license) | `. .\configs\low_vram.ps1; python app_run.py` |
| C. 只读源码学架构 | — | 对照 `vision-ai-inference/docs/05-推理服务v2改造技术方案.html` |

## 显存账(为什么 2GB 够)

模型平时驻 **CPU**,推理才上 GPU、干完迁回;批大小 = `空闲显存×GPU_MEM_SAFETY ÷ 每片激活` 自动缩。

| 模型 | 权重上 GPU | 每片激活 | 2GB 卡实际批 |
|---|---|---|---|
| TinyUNet(玩法 A,tile 256) | ~2MB | 256²×96×2B ≈ 12MB | 8(天花板),占用 <300MB |
| EfficientViT-SAM(玩法 B,tile 512) | ~200MB(fp32) | 512²×96×2B ≈ 50MB | 自动缩到 4~8 |

玩法 A 无 halcon 依赖链(关 REGION_ALIGN 后,门控退化为朴素分区 + 告警,判定逻辑不变);量尺寸(毫米)依赖 calib 目录与 halcon,缺失时轮廓照给、measurements 置零(代码内建的保护性兜底)。

## 与 train_mini 闭环

```
datasets/{产品}/{点位}/ (*.bmp + *.tmp + val/ + {基准图}.json)   ← define_masks 标注
        │
        ▼  ../train_mini:  uv run…/ python main.py --product X
models/{产品}/best_int{交集}_ep{轮}.pth + deploy/(manifest v2 契约)
        │  整目录拷入 {SUPERVISED_CKPT_DIR}/{产品}/
        ▼  本服务
POST /detect → {is_ok, details:[{defect_class, global_contours, measurements}]}
```

ckpt 键(`weight/step/epoch/intersection/hit_defects/quality_score`)、manifest 字段
(`weight/tile_size/threshold/points.{pid}.{ref_name,json,ref_image,zones}`)、best 命名——
train_mini 与真训练工程**逐字段一致**,所以本服务对两者通用(只差模型构造器切换)。

## 目录与 v2 的差异清单

新增 `mini_model.py`、`configs/low_vram.ps1`、本 README;其余文件与
`vision-ai-inference-v2` 完全一致(详见其 README)。学习时改这里的参数不影响真工程。
