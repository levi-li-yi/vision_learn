# Stage 11：指纹增量缓存与监控渲染学习工程（1~2 周）

为读懂 `supervised_apps/train/src/train/prep_plan.py + web.py（Renderer 段）+
runtime_state.py（归档段）+ main.py（清场/子进程段）`——训练从 0→1 的缓存复用
体系与监控页的"按需推理渲染"——而设。前置：无深度学习依赖（本阶段纯工程化，
numpy + 标准库）；建议先过 `train_mini/runtime_state.py + web.py`（轻量版监控）。

## 工程结构

```
stage11_cache_monitor/
├── stage11/
│   ├── fingerprint_plan.py   两档指纹(元数据/SHA256) / 版本号失效 / 三态执行计划 / 原子更新 / 归档轮换
│   ├── render_cache.py       三层 LRU(模型 mtime 失效级联/热力图/底图) / 阈值只重画 / 悬停去抖
│   └── proc_manage.py        TRAIN_WORKER 双角色 / 状态文件竞态 / 优雅退出+进程树击杀
├── tests/                    pytest 用例（含真实子进程拉起与关停）
├── conftest.py / requirements.txt（numpy + pytest）
└── README.md
```

## 环境与运行

```bash
cd D:/ytc_robot/vision_learn/stage11_cache_monitor
python -m venv .venv                      # 或复用任意 stage 的 venv（本阶段零重依赖）
.venv/Scripts/python.exe -m pip install -r requirements.txt

.venv/Scripts/python.exe -m stage11.fingerprint_plan
.venv/Scripts/python.exe -m stage11.render_cache
.venv/Scripts/python.exe -m stage11.proc_manage     # 真实拉起子进程并远程关停

.venv/Scripts/python.exe -m pytest tests/ -v
```

## 10 天学习日程（每天 ~1.5 小时）

| 天 | 内容 | 对应主仓库 |
|----|------|-----------|
| 1 | `fingerprint_plan`：元数据档 vs SHA256 档的分工；改 bmp 大小/改 json 内容各看谁失效 | `_bmp_meta` / `_sha256_file` |
| 2 | 三态计划全生命周期：full→skip→incremental；产出数不足只补差额 | `build_startup_execution_plan` |
| 3 | 原子更新与断点续跑：第二步崩溃后第一步指纹已落盘实验 | `update_saved_fingerprints` |
| 4 | 版本号失效开关 + 归档轮换（.tmp 分家：可复用前置产物 vs 单会话运行态） | `ALIGN_VER` 等 / `runtime_state.tmp_dir` |
| 5-6 | `render_cache`：改阈值只重画；权重 mtime 变化级联清空旧热力图 | `Renderer._model/_heat` |
| 7 | 三层 LRU 容量实验；悬停查全分辨率值与 120ms 去抖 | `render` / `heat_val` / web_page |
| 8 | `proc_manage`：双角色判别；状态文件跨进程交换（web 父进程零 torch 依赖） | `main.py` TRAIN_WORKER 段 |
| 9 | Windows 竞态现场：os.replace 撞上读者 → 写侧重试 + 读者容错 | `runtime_state`（合并式原子写） |
| 10 | 优雅退出窗口 → 进程树击杀（taskkill /T /F）；合书复述"二跑为什么快" | `MonitorServer.kill_child` |

## 通关标准（全部达成即完成阶段 11）

- [ ] 说清两档指纹各防什么坑（bmp 读不起内容→元数据；json 元数据不可靠→SHA256）
- [ ] 写出三态计划判据与 incremental 的含义（指纹同、产出不足 → 只补差额）
- [ ] 解释版本号常量为什么是"最粗暴也最可靠的失效开关"
- [ ] 说清渲染三层缓存各自的 key/容量/失效条件与 mtime 级联清空
- [ ] 解释"改阈值只重画不重推"为什么成立（阈值是渲染参数不是推理参数）
- [ ] 复述 Windows 状态文件竞态的两个补救（写侧 replace 重试 / 读者容错）
- [ ] 说清优雅退出窗口与进程树击杀的次序（先礼后兵：孙进程孤儿问题）
- [ ] `pytest tests/` 全绿

## 与主仓库的参数级对应

| 本工程 | 主仓库 |
|--------|--------|
| bmp 指纹 = 名字+size+mtime；json = SHA256 | `prep_plan._bmp_meta / _sha256_file` |
| `ALIGN_VER/REAL_PATCH_VER/LORA_VER/SYNTH_VER` | 同名常量 |
| skip / incremental / full 三态 | `build_startup_execution_plan` |
| tmp+`os.replace` 原子更新 | `update_saved_fingerprints` |
| 归档留 3 份 | `.tmp/_archive/` 轮换 |
| 模型 ≤2、mtime 失效、级联清热力图 | `Renderer._model` |
| 热力图 ≤3、底图 ≤6、阈值只重画 | `Renderer._heat/_vbase/render` |
| 悬停全分辨率查值 + 120ms 去抖 | `/api/heatval` + web_page 防抖 |
| TRAIN_WORKER=1 双角色 | `main.py` / `web.py::spawn_child` |
| 优雅窗口 + taskkill /T /F | `request_shutdown` / `kill_child` |
| replace 重试 + 读者容错 | `runtime_state`（合并式原子写） |

## 常见坑

- json 用元数据指纹：内容改了但大小 mtime 没变（编辑器保时间戳）→ 旧缓存被错误复现；
- 指纹在每个子步骤里零散探测：探测成本叠加且中途状态漂移——开局一次性全局计划；
- 忘了级联失效：权重换版后监控页还在展示旧热力图，工程师对着过期图 debug；
- Windows 上 os.replace 撞读者直接 PermissionError 崩训练——写侧必须重试；
- 只杀子进程不杀树：DataLoader worker 孙进程变孤儿占着显存；
- shutdown 请求先杀进程再回 HTTP 响应：网页收到的是连接重置——先回话再动手。

## 下一步

全部 stage（0~11）完成。回读 `supervised_apps/train` 全仓库 + `docs/` 图解手册，
或进入 `infer_lite/`（推理服务侧）巩固交付闭环。
