from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from common.exceptions import (
    CkptNotFound,
    InputImageAmbiguous,
    InputImageNotFound,
    ManifestNotFound,
    PointNotFound,
)
from common.logger import setup_logger
from schemas import DetectType

logger = setup_logger("paths")

SUPERVISED_PREFIX = "sup"

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# v2 训练交付权重命名：best_int{总交集}_ep{轮}.pth（按分数命名，与质量脱钩——
# 选择严禁用字典序（best_int999 > best_int1000），必须数值比较）
_BEST_PATTERN = re.compile(r"best_int(\d+)_ep(\d+)\.pth(?:\.tar)?")

#: v2 判别：产品目录含 deploy/manifest.json 且存在部署权重
MANIFEST_RELPATH = ("deploy", "manifest.json")


@dataclass(frozen=True)
class Resolved:
    """寻址结果：一次 /detect 请求要用到的全部要素。"""

    uid: str
    type: DetectType
    product_code: str
    point_id: str
    task_id: str
    ckpt_path: str
    manifest_path: str
    image_path: str
    manifest: dict        # deploy/manifest.json 全量（weight/tile_size/threshold/points/...）
    tile_size: int
    threshold: float      # manifest 携带阈值（0.5）；引擎侧重被 SEG_THRESHOLD 显式覆盖
    deploy_dir: str
    is_supervised: bool = True

    # 点位配置快捷访问（manifest.points[point_id]）
    @property
    def point_meta(self) -> dict:
        return (self.manifest.get("points") or {}).get(self.point_id) or {}


@dataclass(frozen=True)
class TopoEntry:
    uid: str
    type: DetectType
    product_code: str
    ckpt_filename: str
    ckpt_path: str
    manifest_present: bool


def expand(path: str) -> str:
    """展开 ~ 与环境变量。"""
    return os.path.expandvars(os.path.expanduser(path))


class ModelPathResolver:
    """v2 布局 B（原生无 fixture 层）寻址器。

    布局：{SUPERVISED_CKPT_DIR}/{product_code}/
        ├── best_int{交集}_ep{轮}.pth     ← manifest.weight 权威指针；best_int 数值降序兜底
        └── deploy/
            ├── manifest.json             ← 唯一说明书（weight/tile_size/threshold/points）
            ├── tile_size.txt
            └── {point_id}/{基准图名}.json + {基准图名}.bmp   ← 分区 RLE + 配准参照物
    uid：sup::{product_code}（fixture_code 保留必填、仅透传，不参与寻址）
    """

    def __init__(self, supervised_ckpt_dir: str):
        self.root = Path(supervised_ckpt_dir)

    # ------------------------------------------------------------------ #
    @staticmethod
    def build_uid(product_code: str) -> str:
        return f"{SUPERVISED_PREFIX}::{product_code}"

    def _model_dir(self, product_code: str) -> Path:
        return self.root / product_code

    @staticmethod
    def read_manifest(path: str | Path) -> dict:
        """读 deploy/manifest.json（utf-8-sig 兼容 Windows 工具写出的 BOM）。"""
        with open(path, "r", encoding="utf-8-sig") as f:
            return json.load(f)

    @staticmethod
    def _latest_best_ckpt(model_dir: Path, manifest: dict) -> Optional[str]:
        """权重选择：manifest.weight 为权威指针（随 best 原子重建）；
        不一致/缺失时按 best_int(\\d+) **数值**降序取最大兜底（字典序会错）。
        trainer_state.pth.tar 天然不匹配模式被排除。"""
        name = manifest.get("weight")
        if isinstance(name, str) and name:
            if (model_dir / name).is_file():
                return name
            logger.warning(
                f"manifest.weight={name!r} 不存在或非文件，回退 best_int 数值扫描: {model_dir}"
            )
        cands: list[tuple[tuple[int, int], str]] = []
        try:
            files = os.listdir(model_dir)
        except OSError:
            files = []
        for f in files:
            m = _BEST_PATTERN.fullmatch(f)
            if m and (model_dir / f).is_file():
                cands.append(((int(m.group(1)), int(m.group(2))), f))
        if not cands:
            return None
        cands.sort(reverse=True)
        return cands[0][1]

    @staticmethod
    def _resolve_image(dir_path: Path) -> Path:
        """图片目录内必须恰好一张图（.jpg/.jpeg/.png/.bmp/.tif/.tiff）。"""
        if not dir_path.is_dir():
            raise InputImageNotFound(str(dir_path))
        hits = [p for p in sorted(dir_path.iterdir())
                if p.is_file() and p.suffix.lower() in _IMAGE_EXTS]
        if len(hits) == 0:
            raise InputImageNotFound(str(dir_path))
        if len(hits) > 1:
            raise InputImageAmbiguous(str(dir_path), len(hits))
        return hits[0]

    # ------------------------------------------------------------------ #
    def resolve(self, product_code: str, point_id: str, dir_path: str) -> Resolved:
        uid = self.build_uid(product_code)
        model_dir = self._model_dir(product_code)
        if not model_dir.is_dir():
            raise CkptNotFound(str(model_dir))

        manifest_path = model_dir.joinpath(*MANIFEST_RELPATH)
        if not manifest_path.is_file():
            raise ManifestNotFound(str(manifest_path))
        manifest = self.read_manifest(manifest_path)

        points = manifest.get("points") or {}
        if point_id not in points:
            raise PointNotFound(product_code, point_id)

        weight = self._latest_best_ckpt(model_dir, manifest)
        if weight is None:
            raise CkptNotFound(str(model_dir))
        ckpt_path = str(model_dir / weight)

        image_dir = Path(dir_path)
        image_path = self._resolve_image(image_dir)
        task_id = image_dir.name

        try:
            tile_size = int(manifest.get("tile_size") or 512)
        except (TypeError, ValueError):
            logger.warning(f"manifest.tile_size 非法: {manifest.get('tile_size')!r}，按 512")
            tile_size = 512
        try:
            threshold = float(manifest.get("threshold") or 0.5)
        except (TypeError, ValueError):
            logger.warning(f"manifest.threshold 非法: {manifest.get('threshold')!r}，按 0.5")
            threshold = 0.5

        logger.info(
            f"resolved uid={uid} ckpt={weight} point={point_id} tile={tile_size} "
            f"threshold={threshold} image={image_path.name}"
        )
        return Resolved(
            uid=uid,
            type=DetectType.SUPERVISED,
            product_code=product_code,
            point_id=point_id,
            task_id=task_id,
            ckpt_path=ckpt_path,
            manifest_path=str(manifest_path),
            image_path=str(image_path),
            manifest=manifest,
            tile_size=tile_size,
            threshold=threshold,
            deploy_dir=str(manifest_path.parent),
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def enumerate_topology(root: str | Path) -> list[TopoEntry]:
        """盘点根目录下全部 v2 可部署产品（preflight 用；纯磁盘只读，不加载权重）。

        规则：第 1 层产品目录含 deploy/manifest.json 即候选；权重文件名复用运行时
        同源口径（manifest.weight 权威 + best_int 数值兜底）。无 manifest/无权重的
        目录告警跳过。
        """
        root = Path(root)
        out: list[TopoEntry] = []
        if not root.is_dir():
            return out
        for d in sorted(root.iterdir()):
            if not d.is_dir() or d.name.startswith("."):
                continue
            manifest_path = d.joinpath(*MANIFEST_RELPATH)
            if not manifest_path.is_file():
                logger.warning(f"topology skip（非 v2 布局，缺 deploy/manifest.json）: {d}")
                continue
            try:
                manifest = ModelPathResolver.read_manifest(manifest_path)
            except Exception as e:
                logger.warning(f"topology skip（manifest 解析失败 {e}）: {manifest_path}")
                continue
            weight = ModelPathResolver._latest_best_ckpt(d, manifest)
            if weight is None:
                logger.warning(f"topology skip（无部署权重）: {d}")
                continue
            out.append(
                TopoEntry(
                    uid=ModelPathResolver.build_uid(d.name),
                    type=DetectType.SUPERVISED,
                    product_code=d.name,
                    ckpt_filename=weight,
                    ckpt_path=str(d / weight),
                    manifest_present=True,
                )
            )
        return out
