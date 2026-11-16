"""HighwayScene data I/O: frame loading and the lane-OBB proxy labels.

Ground-truth protocol (the repo's lane-region OBB convention): points inside
the lane OBBs are the foreground class ('O'); points outside are the
background class ('V'); there is no ignore class. Preprocessing follows the
repo baseline: r <= max_range crop.

Requires the external roadside-lidar-background-subtraction repository
(HighwayScene dataset class, ground-truth OBB machinery); the lane-OBB
definitions live in the repo's configs/ground_truth/highwayscene.yaml and are
NOT modified here.

Adapted from scripts/stage3_highwayscene/hs_common.py (paths parameterized).
"""

import sys
from pathlib import Path

import numpy as np
import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


class HSContext:
    """Holds the repo imports, dataset handle, and GT params for one config."""

    def __init__(self, config_path):
        cfg = load_config(config_path)
        p = cfg["paths"]
        self.repo_path = Path(p["repo_path"])
        self.data_root = Path(p["data_path"])
        root = Path(config_path).resolve().parents[1]
        gmm = Path(p["initial_gmm"])
        self.initial_gmm = gmm if gmm.is_absolute() else root / gmm
        gt = Path(p.get("gt_config", "configs/ground_truth/highwayscene.yaml"))
        self.gt_yaml = gt if gt.is_absolute() else self.repo_path / gt
        self.max_range = float(cfg["split"].get("max_range", 100.0))
        self.vert = int(cfg["grid"]["vert"])
        self.horiz = int(cfg["grid"]["horiz"])
        self.sensor = cfg.get("sensor", "ouster")

        sys.path.insert(0, str(self.repo_path / "src"))
        from static_bg_subtraction.datasets.highwayscene import HighwayScene
        from static_bg_subtraction.processing.ground_truth import (
            _compute_fg_mask_from_obbs_ground,
        )
        self._fg_fn = _compute_fg_mask_from_obbs_ground
        self._ds = HighwayScene(str(self.data_root / "test"), sensor=self.sensor)
        with open(self.gt_yaml) as f:
            self._gt = yaml.safe_load(f)[self.sensor]

    def frame_ids(self):
        return sorted(self._ds.frame_ids())

    def load_frame(self, fid: int):
        """Return (xyz float64 (N,3), channel int32 (N,), status (N,)['O'/'V'],
        dist (N,)). status: 'O' = inside a lane OBB (foreground class),
        'V' = outside (background class). r > max_range is cropped."""
        frame = self._ds.get_frame_by_id(fid)
        pts = self._ds.get_points(frame)
        xyz = pts.xyz.astype(np.float64)
        ch = pts.channel_ids.astype(np.int32)
        r = np.linalg.norm(xyz, axis=1)
        keep = r <= self.max_range
        xyz, ch, r = xyz[keep], ch[keep], r[keep]
        fg = self._fg_fn(xyz, self._gt)
        status = np.where(fg, "O", "V")
        return xyz, np.clip(ch, 0, self.vert - 1), status, r

    def load_frames(self, fids):
        """[(fid, xyz, channel, status, dist)]."""
        return [(fid, *self.load_frame(fid)) for fid in fids]
