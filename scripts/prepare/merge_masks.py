"""阶段1 步骤5：合并 V/O/ignore 三类掩码。

对每帧：V-proxy 优先，其次 O-proxy，否则 ignore。
输入：scripts/stage1/v_proxy/frame_*.npz + o_proxy/frame_*.npz
输出：scripts/stage1/masks/frame_{id}.npz（frame_id, status('V'/'O'/'ignore'), v_mask, o_mask, dist）
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from tumtraf_io import PACKAGE_ROOT, paths_from_config  # noqa: E402

BUCKETS = [(0, 20), (20, 30), (30, 40), (40, 50), (50, np.inf)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="目录后缀，如 _val → v_proxy_val/o_proxy_val/masks_val")
    ap.add_argument("--config", default=str(PACKAGE_ROOT / "configs/tumtraf.yaml"))
    args = ap.parse_args()
    base = paths_from_config(args.config).masks_dir.parent
    V_DIR = base / f"v_proxy{args.tag}"
    O_DIR = base / f"o_proxy{args.tag}"
    OUT_DIR = base / f"masks{args.tag}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    v_files = sorted(V_DIR.glob("frame_*.npz"), key=lambda p: int(p.stem.split("_")[1]))
    agg = {s: np.zeros(len(BUCKETS), dtype=np.int64) for s in "VOI"}
    tot = np.zeros(len(BUCKETS), dtype=np.int64)
    t0 = time.time()

    for vf in v_files:
        fid = int(vf.stem.split("_")[1])
        v = np.load(vf)
        o = np.load(O_DIR / f"frame_{fid}.npz")
        assert len(v["v_mask"]) == len(o["o_mask"]), f"帧 {fid} 点数不一致"
        status = np.full(len(v["v_mask"]), "I", dtype="U1")
        status[o["o_mask"]] = "O"
        status[v["v_mask"]] = "V"
        dist = v["dist"]
        np.savez_compressed(OUT_DIR / f"frame_{fid}.npz",
                            frame_id=fid, status=status,
                            v_mask=v["v_mask"], o_mask=o["o_mask"], dist=dist)
        for bi, (lo, hi) in enumerate(BUCKETS):
            sel = (dist >= lo) & (dist < hi)
            tot[bi] += sel.sum()
            for s in "VOI":
                agg[s][bi] += (sel & (status == s)).sum()

    print(f"合并 {len(v_files)} 帧 → {OUT_DIR}（{time.time() - t0:.0f}s）\n")
    print(f"{'桶':<8}{'总点数':>10}{'V':>8}{'O':>8}{'ignore':>8}")
    for bi, (lo, hi) in enumerate(BUCKETS):
        hi_s = f"{hi:g}" if np.isfinite(hi) else "inf"
        n = tot[bi]
        print(f"{lo:g}-{hi_s}m{n:>10}{agg['V'][bi]/n:>8.3f}{agg['O'][bi]/n:>8.3f}{agg['I'][bi]/n:>8.3f}")
    n = tot.sum()
    print(f"{'全局':<8}{n:>10}{agg['V'].sum()/n:>8.3f}{agg['O'].sum()/n:>8.3f}{agg['I'].sum()/n:>8.3f}")


if __name__ == "__main__":
    main()
