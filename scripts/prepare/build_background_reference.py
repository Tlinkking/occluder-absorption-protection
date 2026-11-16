"""阶段1 步骤2：构建 TUMTraf-I s03 静态背景参考点云（独立于 GMM/软仲裁/q_V）。

方法：训练窗（帧 1–600）逐帧剔除 3D 框内点 → 0.1m 体素累积 → 每体素取
"逐帧中位数的时序中位数"作为背景距离。不允许使用任何 GMM 输出。

框剔除：用 OBB 的旋转外接 AABB（half = |R_z(yaw)| @ (size/2 + BOX_INFLATE)），
比无视 yaw 的 AABB 更保守（宁可多删，保证背景干净）；OBB 精确剔除留作敏感性分析。

输出：scripts/stage1/background_reference_s03.npz
  centers (M,3) float32         体素中心坐标
  median_distances (M,) float32 体素背景距离（时序中位数）
  frame_counts (M,) int32       该体素出现过的帧数
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).parent))
from tumtraf_io import PACKAGE_ROOT, paths_from_config  # noqa: E402

VOXEL_SIZE = 0.1          # m
BOX_INFLATE = 0.2         # m，框膨胀（各方向）
MIN_VOXEL_POINTS = 5      # 体素至少在这么多帧里出现过才保留
TRAIN_FRAME_RANGE = (1, 600)

_paths = paths_from_config(PACKAGE_ROOT / "configs/tumtraf.yaml")
CONFIG = _paths.repo_path / "configs/examples/tumtraf_run.yaml"
DATA_ROOT = _paths.data_root
OUT_NPZ = _paths.bg_ref_npz

sys.path.insert(0, str(_paths.repo_path / "src"))
from static_bg_subtraction.config import ConfigManager  # noqa: E402
from static_bg_subtraction.datasets.tumtraf import _read_pcd  # noqa: E402

_KEY_OFF = 1 << 20  # 体素索引偏移（±100k m 范围足够）


def voxel_keys(xyz: np.ndarray) -> np.ndarray:
    idx = np.floor(xyz / VOXEL_SIZE).astype(np.int64) + _KEY_OFF
    return (idx[:, 0] << 42) | (idx[:, 1] << 21) | idx[:, 2]


def key_to_center(key: np.ndarray) -> np.ndarray:
    ix = (key >> 42) - _KEY_OFF
    iy = ((key >> 21) & ((1 << 21) - 1)) - _KEY_OFF
    iz = (key & ((1 << 21) - 1)) - _KEY_OFF
    return (np.stack([ix, iy, iz], axis=1).astype(np.float64) + 0.5) * VOXEL_SIZE


def load_cuboids(label_path: Path):
    """[(uuid, type, center(3), yaw, size(3))]，与 run_inference_save.py 同源解析。"""
    data = json.loads(label_path.read_text())
    out = []
    for frame_obj in data.get("openlabel", {}).get("frames", {}).values():
        for uuid, obj in frame_obj.get("objects", {}).items():
            od = obj.get("object_data", {})
            val = od.get("cuboid", {}).get("val")
            if val is None or len(val) != 10:
                continue
            yaw = R.from_quat(val[3:7]).as_euler("xyz")[2]
            out.append((uuid, od.get("type", "?"), np.array(val[0:3]), yaw, np.array(val[7:10])))
    return out


def aabb_half_extents(yaw: float, size: np.ndarray) -> np.ndarray:
    """OBB（z-yaw 旋转）的轴对齐外接框半宽（含膨胀）。"""
    c, s = abs(np.cos(yaw)), abs(np.sin(yaw))
    half = size / 2.0 + BOX_INFLATE
    return np.array([c * half[0] + s * half[1], s * half[0] + c * half[1], half[2]])


def outside_box_mask(xyz: np.ndarray, cuboids) -> np.ndarray:
    keep = np.ones(len(xyz), dtype=bool)
    for _uuid, _t, center, yaw, size in cuboids:
        half = aabb_half_extents(yaw, size)
        inside = (np.abs(xyz - center) <= half).all(axis=1)
        keep &= ~inside
    return keep


def main() -> None:
    cfg = ConfigManager(CONFIG)
    cfg.override_paths(data_root=DATA_ROOT, output_root=_paths.repo_path / "outputs")
    train_ds = cfg.create_dataset(split="train")
    train_ids = sorted(train_ds.frame_ids())
    assert train_ids[0] >= TRAIN_FRAME_RANGE[0] and train_ids[-1] <= TRAIN_FRAME_RANGE[1]
    print(f"训练窗 {len(train_ids)} 帧 ({train_ids[0]}..{train_ids[-1]})")

    voxel_obs = {}  # key -> list of 逐帧中位数距离
    t0 = time.time()
    total_pts, kept_pts = 0, 0
    for fi, fid in enumerate(train_ids):
        frame = train_ds.get_frame_by_id(fid)
        xyz, _ring = _read_pcd(str(frame.path))
        cuboids = load_cuboids(train_ds._label_dir / (frame.path.stem + ".json"))
        keep = outside_box_mask(xyz, cuboids)
        total_pts += len(xyz); kept_pts += int(keep.sum())
        pk = xyz[keep]
        keys = voxel_keys(pk)
        dist = np.linalg.norm(pk, axis=1)
        med = pd.Series(dist).groupby(keys).median()
        for k, v in med.items():
            voxel_obs.setdefault(k, []).append(v)
        if (fi + 1) % 100 == 0:
            print(f"  {fi + 1}/{len(train_ids)} 帧, {time.time() - t0:.0f}s, 体素 {len(voxel_obs)}")

    print(f"框外点占比 {kept_pts / total_pts:.3f} ({kept_pts}/{total_pts})")

    keys = np.array([k for k, v in voxel_obs.items() if len(v) >= MIN_VOXEL_POINTS], dtype=np.int64)
    dropped = len(voxel_obs) - len(keys)
    centers = key_to_center(keys)
    medians = np.array([np.median(voxel_obs[k]) for k in keys], dtype=np.float32)
    counts = np.array([len(voxel_obs[k]) for k in keys], dtype=np.int32)

    np.savez_compressed(OUT_NPZ, centers=centers.astype(np.float32),
                        median_distances=medians, frame_counts=counts)

    d = medians
    print(f"\n体素总数 {len(voxel_obs)}，丢弃(<{MIN_VOXEL_POINTS}帧) {dropped}，保留 {len(keys)}")
    for lo, hi in [(0, 20), (20, 40), (40, 60), (60, np.inf)]:
        n = int(((d >= lo) & (d < hi)).sum())
        tag = " [稀疏!]" if n < 100 else ""
        hi_s = f"{hi:g}" if np.isfinite(hi) else "inf"
        print(f"  {lo:g}-{hi_s}m: {n} 体素{tag}")
    print(f"已写入 {OUT_NPZ}（{time.time() - t0:.0f}s）")


if __name__ == "__main__":
    main()
