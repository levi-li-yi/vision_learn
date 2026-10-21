from __future__ import annotations


class ServiceException(Exception):
    """业务异常基类, 会被全局处理器转为统一响应体。"""

    def __init__(self, message: str, code: int = 500):
        super().__init__(message)
        self.message = message
        self.code = code


class InputImageNotFound(ServiceException):
    def __init__(self, dir_path: str):
        super().__init__(f"dir {dir_path} 内未找到唯一图像", code=400)


class InputImageAmbiguous(ServiceException):
    def __init__(self, dir_path: str, count: int):
        super().__init__(f"dir {dir_path} 内存在 {count} 张图像, 需唯一", code=400)


class CkptNotFound(ServiceException):
    def __init__(self, path: str):
        super().__init__(f"checkpoint 不存在: {path}", code=404)


class ManifestNotFound(ServiceException):
    def __init__(self, path: str):
        super().__init__(f"deploy/manifest.json 不存在(非 v2 布局产品目录): {path}", code=404)


class PointNotFound(ServiceException):
    def __init__(self, product_code: str, point_id: str):
        super().__init__(
            f"point_id {point_id!r} 不在 manifest.points 内: product={product_code}", code=404
        )


class ModelBuildError(ServiceException):
    def __init__(self, message: str):
        super().__init__(f"模型构建失败: {message}", code=500)


class CkptKeyError(ServiceException):
    def __init__(self, ckpt_path: str, missing_keys):
        keys = ", ".join(missing_keys) if isinstance(missing_keys, (list, tuple)) else str(missing_keys)
        super().__init__(f"checkpoint 缺失必要键 [{keys}]: {ckpt_path}", code=500)


class StateDictMismatch(ServiceException):
    def __init__(self, ckpt_path: str, detail: str):
        super().__init__(f"权重加载不匹配 {ckpt_path}: {detail}", code=500)


class CalibNotFound(ServiceException):
    def __init__(self, path: str):
        super().__init__(f"相机内外参缺失（camera_param.dat / pose.dat）: {path}", code=500)


class TopologyInvalid(ServiceException):
    def __init__(self, message: str):
        super().__init__(message, code=400)
