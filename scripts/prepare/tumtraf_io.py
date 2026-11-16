"""TUMTraf-I data I/O for proxy construction and runner scripts.

Adapted from scripts/stage1/common.py: all hard-coded paths are replaced by a
Paths object built from configs/tumtraf.yaml. Requires the external
roadside-lidar-background-subtraction repository (data loading and config
machinery); see README "External dependency".
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

BOX_INFLATE = 0.2  # m, cuboid inflation (same as background-reference build)

PACKAGE_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Paths:
    repo_path: Path
    data_root: Path
    masks_dir: Path
    bg_ref_npz: Path


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def paths_from_config(config_path) -> Paths:
    cfg = load_config(config_path)
    p = cfg["paths"]
    root = Path(config_path).resolve().parents[1]

    def _resolve(v):
        q = Path(v)
        return q if q.is_absolute() else root / q

    return Paths(
        repo_path=Path(p["repo_path"]),
        data_root=Path(p["data_path"]),
        masks_dir=Path(p["masks_path"]),
        bg_ref_npz=_resolve(p["background_reference"]),
    )


def _repo_imports(paths: Paths):
    sys.path.insert(0, str(paths.repo_path / "src"))
    from static_bg_subtraction.config import ConfigManager
    from static_bg_subtraction.datasets.tumtraf import _read_pcd
    return ConfigManager, _read_pcd


def make_dataset(paths: Paths, split: str):
    ConfigManager, _ = _repo_imports(paths)
    cfg = ConfigManager(paths.repo_path / "configs/examples/tumtraf_run.yaml")
    cfg.override_paths(data_root=paths.data_root,
                       output_root=paths.repo_path / "outputs")
    return cfg.create_dataset(split=split)


def read_pcd(paths: Paths, pcd_path):
    _, _read_pcd = _repo_imports(paths)
    return _read_pcd(str(pcd_path))


def load_cuboids(label_path: Path):
    """[(uuid, type, center(3), yaw, size(3))]; OpenLABEL cuboid
    val=[x,y,z,qx,qy,qz,qw,l,w,h]."""
    data = json.loads(label_path.read_text())
    out = []
    for frame_obj in data.get("openlabel", {}).get("frames", {}).values():
        for uuid, obj in frame_obj.get("objects", {}).items():
            od = obj.get("object_data", {})
            val = od.get("cuboid", {}).get("val")
            if val is None or len(val) != 10:
                continue
            yaw = R.from_quat(val[3:7]).as_euler("xyz")[2]
            out.append((uuid, od.get("type", "?"), np.array(val[0:3]), yaw,
                        np.array(val[7:10])))
    return out


def load_frame(dataset, paths: Paths, fid: int):
    """Return (xyz float64 (N,3), cuboids)."""
    frame = dataset.get_frame_by_id(fid)
    xyz, _ring = read_pcd(paths, frame.path)
    cuboids = load_cuboids(dataset._label_dir / (frame.path.stem + ".json"))
    return xyz, cuboids


def aabb_half_extents(yaw: float, size: np.ndarray,
                      inflate: float = BOX_INFLATE) -> np.ndarray:
    """Axis-aligned bounding half-extents of a z-yaw rotated OBB (inflated)."""
    c, s = abs(np.cos(yaw)), abs(np.sin(yaw))
    half = size / 2.0 + inflate
    return np.array([c * half[0] + s * half[1], s * half[0] + c * half[1],
                     half[2]])


def box_membership(xyz: np.ndarray, cuboids, inflate: float = BOX_INFLATE):
    """(in_box bool[N], box_idx int[N], inside_count int[N]) via rotated-AABB."""
    n = len(xyz)
    box_idx = np.full(n, -1, dtype=np.int32)
    count = np.zeros(n, dtype=np.int32)
    for bi, (_uuid, _t, center, yaw, size) in enumerate(cuboids):
        half = aabb_half_extents(yaw, size, inflate)
        inside = (np.abs(xyz - center) <= half).all(axis=1)
        box_idx[inside & (box_idx < 0)] = bi
        count += inside.astype(np.int32)
    return count > 0, box_idx, count


def ray_aabb_blocked(xyz: np.ndarray, cuboids,
                     inflate: float = BOX_INFLATE) -> np.ndarray:
    """For each point, whether the ray from the origin is blocked by any box
    (AABB) before reaching the point. Slab method, vectorized; sensor origin
    = (0,0,0)."""
    dist = np.linalg.norm(xyz, axis=1)
    blocked = np.zeros(len(xyz), dtype=bool)
    for _uuid, _t, center, yaw, size in cuboids:
        half = aabb_half_extents(yaw, size, inflate)
        bmin, bmax = center - half, center + half
        tmin = np.full((len(xyz), 3), -np.inf)
        tmax = np.full((len(xyz), 3), np.inf)
        miss = np.zeros(len(xyz), dtype=bool)
        for ax in range(3):
            d = xyz[:, ax]
            par = np.abs(d) < 1e-12
            miss |= par & ((0.0 < bmin[ax]) | (0.0 > bmax[ax]))
            inv = np.where(par, 0.0, dist / np.where(par, 1.0, d))
            ta, tb = bmin[ax] * inv, bmax[ax] * inv
            tmin[:, ax] = np.where(par, -np.inf, np.minimum(ta, tb))
            tmax[:, ax] = np.where(par, np.inf, np.maximum(ta, tb))
        te = tmin.max(axis=1)
        tx = tmax.min(axis=1)
        hit = (tx > np.maximum(te, 0.0)) & (te < dist - 1e-6) & ~miss
        blocked |= hit
    return blocked


def load_background(npz_path):
    d = np.load(npz_path)
    return (d["centers"].astype(np.float64),
            d["median_distances"].astype(np.float64), d["frame_counts"])
