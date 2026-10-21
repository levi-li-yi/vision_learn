# 低显存(≤2GB)启动配置 —— 在启动 app_run.py 前执行(PowerShell):
#   . .\configs\low_vram.ps1 ; python app_run.py
#
# 策略(见 README_LEARN.md 显存账):
#   GPU_MEM_SAFETY 0.7  只敢用空闲显存的 70%
#   BATCH_MAX 8         单次 forward 最多 8 片(tile=manifest.tile_size,train_mini 交付为 256,极轻)
#   AMP_DTYPE fp16      老卡兜底;30系/Ampere+ 可改 bf16(更快更稳)
#   注意:动态分块本来就会按空闲显存自动缩批,这里再压天花板作双保险

$env:GPU_MEM_SAFETY = "0.7"
$env:BATCH_MAX       = "8"
$env:AMP_DTYPE       = "fp16"     # Ampere+(GTX16/RTX20 之后)建议改 "bf16"

# ---- 场景开关(按需取消注释) ----
# 场景 A:服务 train_mini 交付的 TinyUNet 权重(不需要构建 vision_supervised wheel)
# $env:SUPERVISED_MODEL_MODULE = "mini_model"
# $env:SUPERVISED_MODEL_BUILDER = "build_model"
#
# 场景 B:学习机没装 mvtec-halcon(无 license)→ 关闭逆向配准,门控用朴素分区(仅告警不改判)
# $env:REGION_ALIGN = "0"
#
# 场景 C:目录不在默认位置
# $env:SUPERVISED_CKPT_DIR = "D:/vision/vision-ai-pretrained/supervised"
# $env:CALIB_DIR           = "D:/vision/vision-ai-pretrained/calib"
