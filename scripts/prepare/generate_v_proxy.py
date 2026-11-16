"""阶段1 步骤3：生成测试窗 V-proxy 掩码。

V-proxy = 框外 ∧ 0.3m 内有背景参考体素 ∧ |点距 - 背景中位距| < 0.3m ∧ 射线未被框截断。
用法：python generate_v_proxy.py [--frames all|827,830,...]
输出：scripts/stage1/v_proxy/frame_{id}.npz（含诊断字段，供 pilot_check 使用）
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
from tumtraf_io import (PACKAGE_ROOT, box_membership, load_background,  # noqa: E402
                        load_frame, make_dataset, paths_from_config,
                        ray_aabb_blocked)

NEIGHBOR_RADIUS = 0.3  # m，背景参考最近体素搜索半径（3D）
DIST_TOL = 0.3         # m，点距与背景中位距的容差
RAY_EPS = 1e-6



def process_frame(dataset, paths, fid: int, tree: cKDTree, bg_med: np.ndarray) -> dict:
    xyz, cuboids = load_frame(dataset, paths, fid)
    dist = np.linalg.norm(xyz, axis=1)
    in_box, box_idx, _cnt = box_membership(xyz, cuboids)
    outside = ~in_box

    # 背景匹配
    bg_nn_diff = np.full(len(xyz), np.nan)
    dd, ii = tree.query(xyz[outside], k=1, distance_upper_bound=NEIGHBOR_RADIUS)
    has_nn = np.isfinite(dd) & (ii < len(bg_med))
    idx_out = np.flatnonzero(outside)
    diff = np.abs(dist[outside] - bg_med[np.clip(ii, 0, len(bg_med) - 1)])
    bg_nn_diff[idx_out[has_nn]] = diff[has_nn]
    matched = np.zeros(len(xyz), dtype=bool)
    matched[idx_out[has_nn]] = diff[has_nn] < DIST_TOL

    # 射线遮挡（只算框外点即可，但全量算开销也可接受；这里只对框外点算）
    ray_blocked = np.zeros(len(xyz), dtype=bool)
    ray_blocked[outside] = ray_aabb_blocked(xyz[outside], cuboids)

    v_mask = outside & matched & ~ray_blocked
    return {
        "frame_id": fid,
        "v_mask": v_mask,
        "in_box": in_box,
        "box_idx": box_idx,
        "ray_blocked": ray_blocked,
        "bg_nn_diff": bg_nn_diff,
        "dist": dist.astype(np.float32),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="all", help="all 或逗号分隔帧 ID")
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--tag", default="", help="输出目录后缀，如 _val → v_proxy_val")
    ap.add_argument("--config", default=str(PACKAGE_ROOT / "configs/tumtraf.yaml"))
    args = ap.parse_args()

    paths = paths_from_config(args.config)
    out_dir = paths.masks_dir.parent / f"v_proxy{args.tag}"

    centers, bg_med, _fc = load_background(paths.bg_ref_npz)
    tree = cKDTree(centers)
    print(f"背景参考 {len(centers)} 体素")

    ds = make_dataset(paths, args.split)
    test_ids = sorted(ds.frame_ids())
    fids = test_ids if args.frames == "all" else [int(x) for x in args.frames.split(",")]
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    nv = []
    for fi, fid in enumerate(fids):
        out = process_frame(ds, paths, fid, tree, bg_med)
        np.savez_compressed(out_dir / f"frame_{fid}.npz", **out)
        nv.append((int(out["v_mask"].sum()), len(out["v_mask"])))
        if (fi + 1) % 25 == 0:
            print(f"  {fi + 1}/{len(fids)} 帧, {time.time() - t0:.0f}s")

    v = sum(a for a, _ in nv); n = sum(b for _, b in nv)
    print(f"V-proxy 总体占比 {v / n:.3f} ({v}/{n})，{time.time() - t0:.0f}s，输出 {out_dir}")


if __name__ == "__main__":
    main()
