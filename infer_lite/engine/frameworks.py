from __future__ import annotations

import importlib
import os

from common.exceptions import ModelBuildError
from common.logger import setup_logger

logger = setup_logger("frameworks")

# --------------------------------------------------------------------------- #
# v2 模型类接入接缝（seam）：外部 wheel 导入全部集中于此，环境变量可覆盖。
#
# 模型类 OfficialEfficientViTSeg 来自共享 wheel vision_supervised==0.2.0
# （efficientvit 精简集 + seg + align8dof + rle，与训练仓同一份实现）。
# v2 无 ImageEncoderArgs 概念——构造签名即 build_model(input_size=tile_size)。
# --------------------------------------------------------------------------- #

SUPERVISED_MODEL_MODULE = os.environ.get("SUPERVISED_MODEL_MODULE", "vision_supervised.seg")
SUPERVISED_MODEL_BUILDER = os.environ.get("SUPERVISED_MODEL_BUILDER", "build_model")


def _import_attr(module_path: str, attr: str):
    try:
        mod = importlib.import_module(module_path)
    except Exception as e:
        raise ModelBuildError(
            f"导入 {module_path} 失败: {e}（请核对 vision_supervised==0.2.0 wheel 已安装/"
            f"SUPERVISED_MODEL_MODULE 配置正确）"
        )
    obj = getattr(mod, attr, None)
    if obj is None:
        raise ModelBuildError(f"{module_path} 中不存在 {attr}")
    return obj


def prewarm() -> None:
    """启动期预热外部导入（segment_anything/efficientvit 首次 import 较慢），失败仅告警。"""
    try:
        importlib.import_module(SUPERVISED_MODEL_MODULE)
        logger.info(f"prewarm ok: {SUPERVISED_MODEL_MODULE}")
    except Exception as e:
        logger.warning(f"prewarm {SUPERVISED_MODEL_MODULE} 失败（将在首请求重试）: {e}")


def build_supervised_segmenter(input_size: int):
    """构造 v2 监督分割模型：build_model(input_size=manifest.tile_size)。

    模型恒 3 通道（灰度×3），stages[3..4] 的 LoRA 旁路在构造期自动挂载
    （lora_a/lora_b 键随 state_dict 走，strict 载入需与构造严格一致）。
    """
    builder = _import_attr(SUPERVISED_MODEL_MODULE, SUPERVISED_MODEL_BUILDER)
    try:
        return builder(input_size=input_size)
    except Exception as e:
        raise ModelBuildError(f"{SUPERVISED_MODEL_MODULE}.{SUPERVISED_MODEL_BUILDER} 构造失败: {e}")
