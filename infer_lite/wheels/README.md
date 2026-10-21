# wheels/ —— 本地 wheel 缓存(.whl 不进 Git)

- 部署宿主机为 **Windows**,大包均为 `win_amd64` 构建;产线完整缓存推荐放项目外 `E:\.wheels`。
- 本工程需要放入的 wheel:
  - **`vision_supervised-0.2.0-py3-none-any.whl`**(自研共享包:efficientvit 精简集 + seg + align8dof + rle;
    源码在本仓 `vendor/vision_supervised-0.2.0/`,用 `vendor/build_wheel.ps1` 构建产出后拷入此处)
  - CUDA/编译大包:`torch==2.7.1+cu128`、`torchvision==0.22.1+cu128`、`segment-anything==1.0`、
    `opencv-python==4.13.0.92`、`mvtec-halcon==24111.0.0`(需合法 HALCON license)
- 版本须与构建环境一致(Python 3.10.x,CUDA 12.8,与训练仓 conda env `cv` 同源)。
- **v2 已不需要**:`vision_unsupervised`、`vision_evs`、`flash_attn`(GitHub-only 大包解除)、
  `timm`、`bitsandbytes`。
- 安装(不推荐直接 `pip install -r`,Windows 部署机 find-links 命中被解析层丢弃):
  宿主 Python 3.10 建项目内 `.venv` → 遍历安装
  `pip install --no-build-isolation (Get-ChildItem "E:\.wheels\*.whl") -i 阿里云镜像`;真气隙加 `--no-index`。
- 校验:`python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` 期望 `2.7.1+cu128 True`。
