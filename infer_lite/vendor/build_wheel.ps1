# 构建共享 wheel：vision_supervised==0.2.0（训练/推理同一份实现）
# 用法：在部署构建机（Python 3.10）上执行
#   cd vendor\vision_supervised-0.2.0
#   ..\build_wheel.ps1            # 产出 dist/vision_supervised-0.2.0-py3-none-any.whl
# 构建产物拷入服务工程 wheels/ 与产线缓存 E:\.wheels，随依赖遍历安装。
$ErrorActionPreference = "Stop"
Set-Location "$PSScriptRoot\vision_supervised-0.2.0"
python -m pip install --no-build-isolation --quiet build 2>$null
python -m build --wheel
Write-Host "wheel 已生成: dist\ 下，请拷贝至服务工程 wheels\ 目录"
