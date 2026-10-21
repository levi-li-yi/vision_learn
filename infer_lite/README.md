# Vision AI Inference v2

请求驱动的视觉 AI 推理微服务(FastAPI)——**纯监督**版,对标新训练工程 `supervised_apps/train`(dev@3cdfdcb)的 v2 交付契约。改造方案与逐项依据见姊妹仓库 `vision-ai-inference/docs/05-推理服务v2改造技术方案.html`(v9)。

```
产线: POST /detect(JSON 工单) → 寻址(manifest) → 宿舍取人(LRU/热更新)
      → 锁外备料(灰度裁片 + 有效检测区配准) → 锁内(GPU 前向 + 帐篷窗拼接 + 门控)
      → 锁外(轮廓判类 + HALCON 毫米) → JSON 回单(is_ok + 轮廓 + 测量)
```

## 与 v1(vision-ai-inference)的核心差异

| 主题 | v2 |
|---|---|
| 检测类型 | 仅 `two_d_supervised_detection`(非监督线整线退役) |
| 模型布局 | `{SUPERVISED_CKPT_DIR}/{product}/` = `best_int*.pth` + `deploy/`,**整目录拷入即发布**(布局 B,无 fixture 层) |
| uid | `sup::{product_code}`(fixture_code 保留必填、仅透传) |
| 说明书 | `deploy/manifest.json`(weight/tile_size/threshold/points/zones)+ 各点位分区 json + 基准图 |
| 权重选择 | manifest.weight 权威指针;`best_int(\d+)` **数值**降序兜底(文件名与质量脱钩,字典序会错) |
| 输入 | **灰度 → 3 通道复制 → ImageNet 归一化**(训练同款) |
| 切块 | tile=manifest.tile_size,stride=0.75×tile **全图网格 + 边界 clamp**(考试同款,无 included_coords/无补块) |
| 拼接 | **帐篷权重窗加权平均**(与训练考试逐像素同口径,替代 scatter-max) |
| 门控/配准 | 训练同款共享 wheel:**Point8DofAligner 解 H → 逆向投射 + ±10px 吸附 + 圆滑 = 有效检测区**(smooth_for_infer=True);失败四情形 → 基准系朴素分区 + warn_once(与训练考卷回退同款) |
| 阈值 | manifest.threshold(0.5)为准;SEG_THRESHOLD 显式设置才覆盖 |
| defect_class | 命中轮廓 → 所在有效检测区分图(重叠 argmax)→ manifest 缺陷名 `" / "` 拼接;不落分区兜底 "NG";全被直径过滤降级 "OK" |
| 热更新 | 签名 **(文件名, mtime, size)**——同名不同内容的重训产物靠 mtime 触发重载 |
| 模型类 | `vision_supervised==0.2.0` 共享 wheel(vendored efficientvit 精简集 + OfficialEfficientViTSeg + align8dof + rle;**源码在 `vendor/`,构建后进 wheels/**) |

保留的骨架:模型注册中心(LRU/CPU 常驻/acquire_gpu try/finally)、全局串行推理锁、锁外 IO 与量尺、动态分块(BATCH_MAX/GPU_MEM_SAFETY)、统一异常响应、HALCON 毫米测量与直径过滤、preflight/unload。

## 目录结构

```
vision-ai-inference-v2/
├── app_run.py                # uvicorn 入口(0.0.0.0:8090)
├── core.py                   # AppCore.create_app: lifespan 预热 + 路由 + 异常处理
├── common/                   # config(环境变量) / exceptions / logger
├── schemas/                  # Pydantic 工单/回单/模型状态/preflight
├── engine/
│   ├── paths.py              # 布局 B 寻址:manifest + best_int 数值兜底 + 拓扑盘点
│   ├── registry.py           # LRU 宿舍 + (路径,mtime,size) 签名热更新 + acquire_gpu
│   ├── frameworks.py         # 共享 wheel 导入接缝(SUPERVISED_MODEL_MODULE 可覆盖)
│   ├── loaders.py            # 构造 + weights_only 载入 + strict 校验
│   ├── calib.py              # 相机内外参缓存(HALCON,按产品)
│   ├── defect_size.py        # 像素→世界平面毫米(HALCON)
│   └── inference/
│       ├── base.py           # 模板方法:锁外备料→锁内前向/GPU后处理→锁外收尾 + [CT] 计时
│       └── supervised.py     # 灰度×3/全图网格/帐篷窗/有效检测区/defect_class/毫米过滤
├── api/                      # deps + routers(health / detect / models)
├── vendor/vision_supervised-0.2.0/   # 📦 共享 wheel 源码(构建用)
│   ├── efficientvit/         # ✅ 训练仓 vendored 官方库(已瘦身精简集)
│   ├── vision_supervised/seg.py        # ✅ OfficialEfficientViTSeg + LoRA
│   ├── vision_supervised/align8dof.py  # ✅ Point8DofAligner + 有效检测区三件套
│   └── vision_supervised/rle.py        # ✅ rle_to_mask / mask_to_rle
├── wheels/                   # .whl 缓存(不进 Git;含自研 wheel 构建产物)
└── requirements.txt
```

## 模型文件布局(训练交付物直接拷入)

```
{SUPERVISED_CKPT_DIR}/{product_code}/          # 一产品一目录,拷入即发布
├── best_int{总交集}_ep{轮}.pth                 # 唯一权重(EMA;manifest.weight 指向它)
└── deploy/
    ├── manifest.json                          # {weight, tile_size, threshold:0.5,
    │                                          #  points:{pid:{ref_name,json,ref_image,zones:{分区:[缺陷名]}}}}
    ├── tile_size.txt
    └── {point_id}/{基准图名}.json + {基准图名}.bmp   # 分区 RLE(含 domain) + 配准参照物
```

相机内外参(量尺寸用):`{CALIB_DIR}/{product_code}/camera_param.dat + pose.dat`(一机种一套,现场示教固化)。

## 配置(环境变量)

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `SUPERVISED_CKPT_DIR` | `D:/vision/vision-ai-pretrained/supervised` | 模型根目录(布局 B) |
| `CALIB_DIR` | `D:/vision/vision-ai-pretrained/calib` | 相机内外参根目录 |
| `MODEL_NUM_LIMIT` | `100` | 注册中心驻留上限(LRU) |
| `DEVICE` | `cuda` | 推理设备 |
| `AMP_DTYPE` | `bf16` | 精度档 bf16\|fp16\|fp32(bf16 需 SM80+;旧 HALF_PRECISION 兼容) |
| `SEG_THRESHOLD` | 未设置 | **v2 语义**:未设置=用 manifest.threshold(0.5);显式设置=覆盖 |
| `BATCH_MAX` | `24` | 单次 forward 最大片数 |
| `GPU_MEM_SAFETY` | `0.75` | 动态分块显存安全系数(6GB 卡建议 0.7) |
| `MIN_DEFECT_DIAMETER_MM` | `0.5` | 直径过滤阈值(mm),小于此值的轮廓连同测量一并丢弃 |
| `REGION_ALIGN` | `1` | 有效检测区配准总闸(0=朴素分区蒙版;失败情形一律回退朴素+告警) |
| `SUPERVISED_MODEL_MODULE` / `SUPERVISED_MODEL_BUILDER` | `vision_supervised.seg` / `build_model` | 模型构造接缝覆盖 |

HOST/PORT(0.0.0.0/8090)为代码内默认,无环境变量。

## 运行

```bash
# 1) 构建并放置共享 wheel(一次性)
cd vendor && pwsh build_wheel.ps1   # 产出 vision_supervised-0.2.0-py3-none-any.whl → 拷入 wheels/
# 2) 建 .venv 并遍历安装 wheels(见 wheels/README.md)
# 3) 启动
python app_run.py                   # 或 uvicorn app_run:app --host 0.0.0.0 --port 8090
```

日志:`./logs/*.log`(20MB 轮转×5 + stdout);每请求一行 `[CT]` 耗时(lock_ms 为吞吐核心指标)。

## API

- `POST /detect`:请求体 8 字段(type 仅监督值);`dir` 内**有且仅有一张图**;`point_id` 必在 manifest.points。响应 `{code,message,data:{is_ok,details:[{task_id,type,defect_class,global_contours,measurements}]}}`;`global_contours` 与 `measurements` 按序等长;`is_ok = 全部 detail 为 OK`。
- `GET /health` / `GET /models` / `GET /models/{uid}` / `GET /models/resolve?product_code=…`
- `POST /models/preflight`:`{"dir":"~/release"}`(候选根须含 `supervised/`);四态清单 + 疑似降级(quality_score 优先,best_int 数值退化)。
- `POST /models/unload`:`{"uid":"sup::{product}"}`。

异常统一转 `{code, message, data:null}`:400(InputImageNotFound/Ambiguous/TopologyInvalid)、404(CkptNotFound/ManifestNotFound/PointNotFound)、500(ModelBuildError/CkptKeyError/StateDictMismatch/CalibNotFound)。

## 发布流程

训练交付 `models/{产品}/` → `POST /models/preflight` 点名 → **整目录拷入** `{SUPERVISED_CKPT_DIR}/{产品}/`(先拷新后删旧;**拷贝不得保留源 mtime**,否则热更新签名失效)→ 下一请求 JIT 自动换新权重,服务不重启。

## 验收清单(上线前逐项过,对应方案 05 第 12 章)

1. 同一 val 考卷:本服务热力图与训练 `predict_heatmap` 逐像素一致;2. 0.5 阈值 OK/NG 与考试一致;3. 有效检测区与训练 align.json 的 val 图 zones_rle 逐像素一致;4. H 与训练 aligncache 逐元素一致(种子 42);5. align_and_compute_effective_zones 输出与训练同函数一致;6. defect_class 五类用例(单区单缺陷/单区多缺陷拼接/跨区并集/全滤→OK/兜底 NG);7. 缺件 400/404 无 500;8. 同名不同 mtime 覆盖必换新;9. LoRA strict 载入 + bf16 前向 + tile≠512 复测;10. 旧依赖残留清零(无 flash_attn/vision_evs/vision_unsupervised);11. 88 片 lock_ms 达标 + 配准/有效区 ≤200ms(锁外);12. preflight 四态 + 疑似降级。

## 已知约束

- 训练工程契约基线为 **origin/dev @ 3cdfdcb(尚未合入 main)**——若最终以 main 为准,门控回退朴素投影口径(wheel 兼容,仅少调 compute_effective_zone 一步)。
- 高并发下串行锁为吞吐天花板(业务已确认可接受)。
- HALCON 需合法 license(配准 + 量尺寸双消费)。
