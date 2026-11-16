"""阶段1 步骤4：生成测试窗 O-proxy 掩码。

O-proxy = 框内 ∧ 非框边界 ∧ 非重叠框 ∧ 射线方向后方存在更远的背景参考体素。
"后方背景"用方位/俯仰角栅格索引背景参考实现：对点 p 的 (az, el)，
在 ±2 个角栅格邻域内找最大 median_distance，若 > dist(p) + DELTA_R_MIN 则成立。

用法：python generate_o_proxy.py [--frames all|827,830,...]
输出：scripts/stage1/o_proxy/frame_{id}.npz（含诊断字段）
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from tumtraf_io import (BOX_INFLATE, PACKAGE_ROOT, aabb_half_extents,  # noqa: E402
                        box_membership, load_background, load_frame,
                        make_dataset, paths_from_config)

DELTA_R_MIN = 1.0     # m，当前点到后方背景的最小距离差
BOX_BOUNDARY = 0.2    # m，距框面小于此值视为边界点 → ignore
AZ_BIN_DEG = 0.2      # 背景角栅格方位 bin（Ouster 水平分辨率 ~0.18°）
EL_BIN_DEG = 0.5      # 背景角栅格俯仰 bin
NB_BIN = 2            # 邻域 ±2 bin（≈±0.4° 方位、±1.0° 俯仰）



def build_angle_grid(centers: np.ndarray, med: np.ndarray) -> dict:
    """(iaz, iel) -> 该角栅格内最大背景中位距离。"""
    r = np.linalg.norm(centers, axis=1)
    az = np.degrees(np.arctan2(centers[:, 1], centers[:, 0]))
    el = np.degrees(np.arcsin(np.clip(centers[:, 2] / r, -1, 1)))
    iaz = np.floor(az / AZ_BIN_DEG).astype(np.int64)
    iel = np.floor(el / EL_BIN_DEG).astype(np.int64)
    grid = {}
    for k, v in zip(zip(iaz, iel), med):
        if k not in grid or v > grid[k]:
            grid[k] = v
    return grid


def behind_background_max(xyz: np.ndarray, grid: dict) -> np.ndarray:
    """每个点的射线方向后方最大背景距离（无背景则 NaN）。"""
    r = np.linalg.norm(xyz, axis=1)
    az = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))
    el = np.degrees(np.arcsin(np.clip(xyz[:, 2] / np.maximum(r, 1e-9), -1, 1)))
    iaz = np.floor(az / AZ_BIN_DEG).astype(np.int64)
    iel = np.floor(el / EL_BIN_DEG).astype(np.int64)
    out = np.full(len(xyz), np.nan)
    for da in range(-NB_BIN, NB_BIN + 1):
        for de in range(-NB_BIN, NB_BIN + 1):
            vals = np.array([grid.get((a + da, e + de), np.nan) for a, e in zip(iaz, iel)])
            out = np.fmax(out, vals)
    return out


def process_frame(dataset, paths, fid: int, grid: dict) -> dict:
    xyz, cuboids = load_frame(dataset, paths, fid)
    dist = np.linalg.norm(xyz, axis=1)
    in_box, box_idx, count = box_membership(xyz, cuboids)

    # 边界：点距所属框（膨胀后）任意面的最小间隙 < BOX_BOUNDARY → 边界点
    boundary = np.zeros(len(xyz), dtype=bool)
    for bi, (_u, _t, center, yaw, size) in enumerate(cuboids):
        sel = box_idx == bi
        if not sel.any():
            continue
        half = aabb_half_extents(yaw, size, BOX_INFLATE)
        gap = (half - np.abs(xyz[sel] - center)).min(axis=1)
        boundary[sel] = gap < BOX_BOUNDARY

    overlap = count >= 2

    behind_max = np.full(len(xyz), np.nan)
    behind_max[in_box] = behind_background_max(xyz[in_box], grid)
    has_behind = behind_max > dist + DELTA_R_MIN

    o_mask = in_box & ~boundary & ~overlap & has_behind
    return {
        "frame_id": fid,
        "o_mask": o_mask,
        "in_box": in_box,
        "o_boundary": boundary,
        "o_overlap": overlap,
        "behind_max": behind_max,
        "dist": dist.astype(np.float32),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="all", help="all 或逗号分隔帧 ID")
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--tag", default="", help="输出目录后缀，如 _val → o_proxy_val")
    ap.add_argument("--config", default=str(PACKAGE_ROOT / "configs/tumtraf.yaml"))
    args = ap.parse_args()

    paths = paths_from_config(args.config)
    out_dir = paths.masks_dir.parent / f"o_proxy{args.tag}"

    centers, bg_med, _fc = load_background(paths.bg_ref_npz)
    grid = build_angle_grid(centers, bg_med)
    print(f"背景角栅格 {len(grid)} bin")

    ds = make_dataset(paths, args.split)
    test_ids = sorted(ds.frame_ids())
    fids = test_ids if args.frames == "all" else [int(x) for x in args.frames.split(",")]
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    no, nb = 0, 0
    for fi, fid in enumerate(fids):
        out = process_frame(ds, paths, fid, grid)
        np.savez_compressed(out_dir / f"frame_{fid}.npz", **out)
        no += int(out["o_mask"].sum()); nb += int(out["in_box"].sum())
        if (fi + 1) % 25 == 0:
            print(f"  {fi + 1}/{len(fids)} 帧, {time.time() - t0:.0f}s")

    print(f"O-proxy 占框内点比例 {no / max(nb, 1):.3f} ({no}/{nb})，{time.time() - t0:.0f}s，输出 {out_dir}")


if __name__ == "__main__":
    main()
