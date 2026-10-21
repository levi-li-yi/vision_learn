"""vision_supervised 0.2.0 —— 训练/推理共享的监督模型与逆向对齐原语包。

内容四件(全部提取自训练仓 supervised_apps/train,详见各模块头注释):
  · efficientvit/    vendored 官方 EfficientViT(训练已瘦身为精简集)
  · seg.py           OfficialEfficientViTSeg + CascadedSegHead + LoRA 挂载
  · align8dof.py     Point8DofAligner + project_mask_to_query + 有效检测区三件套
  · rle.py           rle_to_mask / mask_to_rle

与旧版 vision_supervised==0.1.0(旧 EfficientViTSeg, 连带 flash_attn)仅同名、
内容完全不同; 两侧严禁私自改动参数——口径漂移即判定失效。
"""

__version__ = "0.2.0"


def __getattr__(name):  # 惰性导出: 避免 import 包即拉起 torch/segment_anything
    if name == "build_model":
        from vision_supervised.seg import build_model
        return build_model
    if name in ("Point8DofAligner", "project_mask_to_query",
                "compute_effective_zone", "align_and_compute_effective_zones",
                "SNAP_PX"):
        from vision_supervised import align8dof
        return getattr(align8dof, name)
    if name in ("rle_to_mask", "mask_to_rle"):
        from vision_supervised import rle
        return getattr(rle, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
