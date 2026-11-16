"""生成补充材料 Fig S2 的两张代理标注着色点云示意图（PDF 矢量 + 300dpi PNG 备用）。

输入（只读）：
  - TUMTraf-I：点云经 scripts/stage1/common.py 的 load_frame（repo _read_pcd），
    掩码 scripts/stage1/masks/frame_1004.npz（字段 status: 'V'/'O'/'I'，
    与点云点索引一一对齐，stage2_clean/runner.py:load_frames 有 assert 保证），
    3D 框来自 TUMTraf-I OpenLABEL（stage1/common.py:load_cuboids）。
  - HighwayScene：点云与掩码经 scripts/stage3_highwayscene/hs_common.py 的
    load_frame（status: 'O'=车道 OBB 内前景 / 'V'=背景，无 ignore），
    lane-OBB 参数来自 roadside-lidar-background-subtraction/configs/ground_truth/highwayscene.yaml。

帧选择逻辑（不改数据，只选展示帧）：
  - TUMTraf-I frame 1004：测试段 931–1033 中 O-proxy 点最多的帧
    （3920 个 O 点，占 12.5%，总 31378 点；统计自 stage1/masks 全量扫描）。
  - HighwayScene frame 5710：抽稀日志中 foreground 比例最接近 15% 的帧
    （15.0%，6672 个前景点；统计自 stage4_ablation/logs_A2_highwayscene.npz，
    label==1 为前景，见 fig4_highwayscene_of1_data.json 的口径说明）。

输出：
  - upload/02_figures/figures/figS2a_tumtraf_proxy_labels.{pdf,png}
  - upload/02_figures/figures/figS2b_highwayscene_laneobb.{pdf,png}
  散点层 rasterized（300 dpi）嵌入 PDF，文字/图例/比例尺/线框保持矢量，
  避免 3 万+ 点的纯矢量 PDF 阅读器卡顿。

运行：conda activate ch2_lidar && python scripts/figures/make_proxy_example_figures.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "scripts/prepare"))

OUT_DIR = Path(__file__).resolve().parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

C_V, C_O, C_IGN = "#2ca02c", "#d62728", "#cccccc"
FIGSIZE = (3.5, 2.8)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.grid": False,
})


def cuboid_edges(center, yaw, size):
    """z-yaw 旋转长方体的 12 条边（segment 列表）。"""
    l, w, h = size / 2.0
    corners = np.array([[x, y, z] for x in (-l, l) for y in (-w, w) for z in (-h, h)])
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    pts = corners @ rot.T + center
    idx = [(0,1),(0,2),(1,3),(2,3),(4,5),(4,6),(5,7),(6,7),(0,4),(1,5),(2,6),(3,7)]
    return [(pts[i], pts[j]) for i, j in idx]


def obb_edges_rpy(center, size, rpy_deg, bottom_only=False):
    """HS yaml 的 OBB（roll/pitch/yaw 度数）边；bottom_only=True 时只给底面 4 条边。"""
    from scipy.spatial.transform import Rotation as R
    l, w, h = np.asarray(size, dtype=float) / 2.0
    corners = np.array([[x, y, z] for x in (-l, l) for y in (-w, w) for z in (-h, h)])
    pts = R.from_euler("xyz", rpy_deg, degrees=True).apply(corners) + np.asarray(center)
    idx = [(0,2),(0,4),(2,6),(4,6)] if bottom_only else [
        (0,1),(0,2),(1,3),(2,3),(4,5),(4,6),(5,7),(6,7),(0,4),(1,5),(2,6),(3,7)]
    return [(pts[i], pts[j]) for i, j in idx]


def style_ax(ax, xlim, ylim, zlim, title_none=True):
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)
    ax.set_xlabel("X (m)", labelpad=1); ax.set_ylabel("Y (m)", labelpad=1)
    ax.set_zlabel("Z (m)", labelpad=1)
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor("#dddddd")
    ax.tick_params(labelsize=7, pad=0)


def add_scale_bar(ax, start, length=10.0, direction=(1, 0, 0), label="10 m"):
    start = np.asarray(start, dtype=float)
    end = start + length * np.asarray(direction, dtype=float)
    ax.plot(*zip(start, end), color="black", lw=1.2, solid_capstyle="butt")
    mid = (start + end) / 2
    ax.text(mid[0], mid[1], mid[2] + 0.8, label, fontsize=8, ha="center")


def save(fig, stem):
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        p = OUT_DIR / f"{stem}.{ext}"
        fig.savefig(p, bbox_inches="tight", **kw)
        print(f"  wrote {p} ({p.stat().st_size/1024:.0f} KiB)")
    plt.close(fig)


def plot_tumtraf(config_path):
    from tumtraf_io import load_frame, make_dataset, paths_from_config

    fid = 1004
    paths = paths_from_config(config_path)
    ds = make_dataset(paths, "test")
    xyz, cuboids = load_frame(ds, paths, fid)
    m = np.load(paths.masks_dir / f"frame_{fid}.npz")
    status = m["status"]
    assert len(status) == len(xyz), "mask/point-count mismatch"

    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111, projection="3d")
    xlim, ylim, zlim = (-35, 35), (-30, 30), (-4, 8)
    for st, color, size, alpha, zorder in (
        ("I", C_IGN, 0.3, 0.25, 1), ("V", C_V, 0.3, 0.9, 2), ("O", C_O, 1.5, 1.0, 3),
    ):
        sel = status == st
        ax.scatter(xyz[sel, 0], xyz[sel, 1], xyz[sel, 2], s=size, c=color,
                   alpha=alpha, linewidths=0, rasterized=True, zorder=zorder)

    n_box = 0
    for _uuid, _type, center, yaw, size in cuboids:
        if not (xlim[0] < center[0] < xlim[1] and ylim[0] < center[1] < ylim[1]):
            continue
        segs = cuboid_edges(center, yaw, size)
        ax.add_collection3d(Line3DCollection(segs, colors="#444444", lw=0.5))
        n_box += 1

    style_ax(ax, xlim, ylim, zlim)
    ax.set_zticks([-4, 0, 4, 8])
    ax.set_box_aspect((70, 60, 12))
    ax.view_init(elev=28, azim=-60)
    add_scale_bar(ax, (-33, 26, -4), length=10.0, direction=(1, 0, 0))
    ax.legend(handles=[
        Line2D([0], [0], marker="s", ls="", ms=6, mfc=C_V, mec=C_V, label="V (background proxy)"),
        Line2D([0], [0], marker="s", ls="", ms=6, mfc=C_O, mec=C_O, label="O (foreground proxy)"),
        Line2D([0], [0], marker="s", ls="", ms=6, mfc=C_IGN, mec=C_IGN, label="ignore"),
    ], loc="upper right", fontsize=7, framealpha=0.9, borderpad=0.3, handletextpad=0.2)

    n = {s: int((status == s).sum()) for s in "VOI"}
    print(f"TUMTraf-I frame {fid}: total={len(status)} V={n['V']} ({n['V']/len(status)*100:.1f}%) "
          f"O={n['O']} ({n['O']/len(status)*100:.1f}%) ignore={n['I']} ({n['I']/len(status)*100:.1f}%) "
          f"3D-boxes drawn={n_box}, view=(elev=28, azim=-60)")
    save(fig, "figS2a_tumtraf_proxy_labels")


def plot_highwayscene(config_path):
    from highwayscene_io import HSContext

    fid = 5710
    ctx = HSContext(config_path)
    xyz, _ch, status, _dist = ctx.load_frame(fid)

    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111, projection="3d")
    xlim, ylim, zlim = (-45, 55), (-30, 20), (-5, 10)
    for st, color, size, zorder in (("V", C_V, 0.5, 2), ("O", C_O, 3.0, 3)):
        sel = status == st
        ax.scatter(xyz[sel, 0], xyz[sel, 1], xyz[sel, 2], s=size, c=color,
                   linewidths=0, rasterized=True, zorder=zorder)

    n_obb = 0
    z_floor = zlim[0] + 0.3  # OBB 底面轮廓投影到地面平面，避免被点云遮挡
    for bb in ctx._gt["bboxes"]:
        segs = obb_edges_rpy(bb["center"], bb["size"], bb["rpy_deg"], bottom_only=True)
        segs = [(np.array([a[0], a[1], z_floor]), np.array([b[0], b[1], z_floor]))
                for a, b in segs]
        ax.add_collection3d(Line3DCollection(
            segs, colors="#666666", lw=0.5, linestyles="dashed", zorder=1))
        n_obb += 1

    style_ax(ax, xlim, ylim, zlim)
    ax.set_zticks([-5, 0, 5, 10])
    ax.set_box_aspect((100, 50, 15))
    ax.view_init(elev=32, azim=-72)
    add_scale_bar(ax, (-43, -27, -5), length=10.0, direction=(1, 0, 0))
    ax.legend(handles=[
        Line2D([0], [0], marker="s", ls="", ms=6, mfc=C_O, mec=C_O, label="Foreground (in lane OBB)"),
        Line2D([0], [0], marker="s", ls="", ms=6, mfc=C_V, mec=C_V, label="Background"),
    ], loc="upper right", fontsize=7, framealpha=0.9, borderpad=0.3, handletextpad=0.2)

    n_o, n_v = int((status == "O").sum()), int((status == "V").sum())
    print(f"HighwayScene frame {fid}: total={len(status)} fg={n_o} ({n_o/len(status)*100:.1f}%) "
          f"bg={n_v} ({n_v/len(status)*100:.1f}%) lane-OBBs drawn={n_obb}, view=(elev=32, azim=-72)")
    save(fig, "figS2b_highwayscene_laneobb")


if __name__ == "__main__":
    plot_tumtraf(PACKAGE_ROOT / "configs/tumtraf.yaml")
    plot_highwayscene(PACKAGE_ROOT / "configs/highwayscene.yaml")
