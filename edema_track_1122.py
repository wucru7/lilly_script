#!/usr/bin/env python3
"""edema_track.py — longitudinal foot-shape tracking for the 62D0 vs 62D3
edema-recovery pair.

Inputs (read-only):
  curvature_62D0L_edema.ply   D0 (day 0, edema)
  curvature_62D3L_healthy.ply D3 (day 3, healthy, recovered)
Both: 44,395 vertices, SAME faces (vertex-wise correspondence), watertight,
metres, left foot + partial lower leg. Leg-cut height differs slightly
between visits.

Convention:
  baseline  = D3 healthy  (reference)
  follow-up = D0 edema    (positive signed displacement = outward from healthy,
                           i.e. the edematous region)

Pipeline (one subject):
  1. Load both PLYs with trimesh (process=False), ignore colors.
  2. Rigid Kabsch alignment on ALL vertices (correspondence is known).
  3. Canonicalisation: centre on baseline centroid, keep raw axes
     (x=length, y=width, z=height), rotate so toes point to +x.
  4. Signed per-vertex displacement along baseline normals (correspondence
     is exact — no ICP / raycast needed).
  5. Non-overlapping geometric partition: heel / midfoot / forefoot, each
     split dorsal vs plantar; big_toe and lesser_toes; front_ankle and
     ankle_other; plus a "leg_cut" zone (top-z vertices) kept out of the
     clinical regions so leg-cut differences don't masquerade as swelling.
  6. Per-region signed-displacement statistics (mean / median / p5 / p95 /
     rms).
  7. Spectral shape latent on each sizeable region (first K=50 graph-
     Laplacian eigenvectors of the region submesh) + a canonical-dorsum
     prototype inner product.

Outputs (written only under /node/scratch/$USER/):
  results.json                     full numeric report
  signed_disp_heatmap.png          4-view baseline mesh coloured by signed
                                   displacement (red = outward = edema)
  region_bars.png                  per-region mean + p5-p95 bars
  signed_disp_mm.npy, ...          raw per-vertex arrays
  baseline_canonical.ply,
  followup_canonical.ply           processed meshes for external tools

Run on the remote server inside the edema venv:
  source /node/scratch/$USER/venvs/edema/bin/activate
  python edema_track.py                            # defaults
  python edema_track.py --baseline <path> --followup <path> --out <dir>
"""

import os
import sys
import json
import argparse
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import trimesh
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Defaults for this specific pair
# ---------------------------------------------------------------------------
DEFAULT_SHARE = ("/node/scratch/l142586/edema_monitoring_share/code/"
                 "compare_62D0_62D3_anklebump_large")
DEFAULT_D0 = os.path.join(DEFAULT_SHARE, "curvature_62D0L_edema.ply")
DEFAULT_D3 = os.path.join(DEFAULT_SHARE, "curvature_62D3L_healthy.ply")
DEFAULT_OUT = f"/node/scratch/{os.environ.get('USER', 'user')}/edema_track_out"


# ---------------------------------------------------------------------------
# Rigid alignment (Kabsch — closed form, correspondence known)
# ---------------------------------------------------------------------------
def kabsch(P, Q):
    """Find R, t such that R @ P.T + t aligns P onto Q (P, Q are N x 3 with
    vertex-wise correspondence). Returns R (3,3), t (3,), per-vertex residual
    norm array (N,) in the aligned frame."""
    mu_P = P.mean(axis=0)
    mu_Q = Q.mean(axis=0)
    H = (P - mu_P).T @ (Q - mu_Q)
    U, _, Vh = np.linalg.svd(H)
    R = Vh.T @ U.T
    if np.linalg.det(R) < 0:
        Vh[-1, :] *= -1
        R = Vh.T @ U.T
    t = mu_Q - R @ mu_P
    aligned = P @ R.T + t
    resid = np.linalg.norm(aligned - Q, axis=1)
    return R, t, resid


# ---------------------------------------------------------------------------
# Canonicalisation: PCA on baseline, same transform applied to both
# ---------------------------------------------------------------------------
def pca_canonicalise_pair(V_base, V_followup):
    """Centre on baseline centroid, rotate by PCA on baseline. Apply the SAME
    transform to both meshes so they stay aligned."""
    c = V_base.mean(axis=0)
    _, _, Vh = np.linalg.svd(V_base - c, full_matrices=False)
    R = Vh
    if np.linalg.det(R) < 0:
        R[2] *= -1
    V_b_c = (V_base - c) @ R.T
    V_f_c = (V_followup - c) @ R.T
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = -R @ c
    return V_b_c, V_f_c, T


def fix_z_up(V_world, N_world, R):
    """If mean outward-facing normal points -z under R, flip z (paired with y
    to preserve right-handedness)."""
    if (N_world @ R.T)[:, 2].mean() < 0:
        return np.diag([1.0, -1.0, -1.0]) @ R
    return R


# ---------------------------------------------------------------------------
# Spectral basis + submesh utilities
# ---------------------------------------------------------------------------
def mesh_graph_laplacian(V, F):
    n = len(V)
    E = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    E = np.sort(E, axis=1); E = np.unique(E, axis=0)
    rows = np.concatenate([E[:, 0], E[:, 1]])
    cols = np.concatenate([E[:, 1], E[:, 0]])
    data = np.ones(len(rows), dtype=np.float64)
    A = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    deg = np.asarray(A.sum(axis=1)).ravel()
    return (sp.diags(deg) - A).tocsc()


def spectral_basis(L, K):
    n = L.shape[0]
    K_eff = min(K, max(2, n - 2))
    try:
        eigvals, Phi = eigsh(L, k=K_eff, sigma=0, which="LM")
    except Exception:
        eigvals, Phi = eigsh(L, k=K_eff, which="SM")
    return Phi[:, np.argsort(eigvals)]


def sub_mesh(V, F, keep_vidx):
    keep_mask = np.zeros(len(V), dtype=bool); keep_mask[keep_vidx] = True
    face_keep = keep_mask[F].all(axis=1)
    Fk = F[face_keep]
    old2new = -np.ones(len(V), dtype=np.int64)
    old2new[keep_vidx] = np.arange(len(keep_vidx))
    return V[keep_vidx], old2new[Fk]


def canonical_prototype_field(V, N, amp_mm=4.0,
                              center=None, sigma_m=0.025):
    """Dorsum-style Gaussian bump used as the matched-filter prototype."""
    if center is None:
        # Use a vertex near the midfoot dorsum apex.
        x_in = np.abs(V[:, 0]) < 0.03
        score = V[:, 2] * np.clip(N[:, 2], 0.0, 1.0)
        score[~x_in] = -np.inf
        center = V[int(np.argmax(score))]
    d2 = np.sum((V - center) ** 2, axis=1)
    radial = np.exp(-d2 / (2.0 * sigma_m ** 2))
    gate = np.clip(N[:, 2], 0.0, 1.0)
    mask = radial * gate
    peak = max(mask.max(), 1e-12)
    return amp_mm * (mask / peak)[:, None] * N


# ---------------------------------------------------------------------------
# Region partition (geometric heuristic since no keypoints supplied)
# ---------------------------------------------------------------------------
_SLICES = ["heel", "midfoot", "forefoot"]
_SURF = ["dorsal", "plantar"]
_ML = ["medial", "lateral"]
REGION_NAMES = ([f"{a}_{b}_{c}" for a in _SLICES for b in _SURF for c in _ML]
                + [f"{a}_{b}" for a in _SLICES for b in _SURF]   # used if ML split is off
                + ["big_toe", "lesser_toes", "front_ankle", "ankle_other"])
# "leg_cut" is also assigned but kept out of REGION_NAMES (artifact zone)


def define_regions(V, N, leg_cut_z_quantile=0.95, ankle_height_frac=0.45,
                   toe_length_m=0.045, big_toe_width_frac=0.38,
                   big_toe_side="auto", front_ankle_nx=0.3,
                   split_medial_lateral=True, n_midline_bins=20):
    """Non-overlapping partition of the baseline vertices. Every vertex gets
    exactly one label from mutually exclusive conditions (no rule overwrites
    another). Frame: x heel->toe, y across, z up, metres.

      leg_cut      z >= leg_cut_z_quantile               (artifact, not clinical)
      front_ankle  ankle band, normal facing forward (n_x > front_ankle_nx)
      ankle_other  ankle band, everything else (sides, back, malleoli)
      big_toe      foot below ankle band, within toe_length_m of the toe tip,
                   on the big-toe side, big_toe_width_frac of the toe width
      lesser_toes  rest of that toe zone
      heel/midfoot/forefoot x {dorsal, plantar} x {medial, lateral}
                   remaining foot, three equal-count slices along x, split by
                   whether the normal points up (dorsal) or down (plantar),
                   then by which side of the foot midline the vertex is on
                   (medial = big-toe side). The midline is the median y in
                   n_midline_bins bins along x, so it follows the foot's curve.

    ankle band = from ankle_height_frac of the way between sole (min z) and
    the leg cut, up to the leg cut.
    """
    x, y, z = V[:, 0], V[:, 1], V[:, 2]
    z_cut = np.quantile(z, leg_cut_z_quantile)
    z_ank = z.min() + ankle_height_frac * (z_cut - z.min())

    leg_cut = z >= z_cut
    ankle_band = (z >= z_ank) & ~leg_cut
    foot = ~leg_cut & ~ankle_band

    toe_zone = foot & (x >= x[foot].max() - toe_length_m)
    yt, zt = y[toe_zone], z[toe_zone]
    y_mid = 0.5 * (yt.min() + yt.max())
    if big_toe_side == "auto":
        # the big toe is the thickest toe: compare height on each side using
        # only the last 15 mm before the tips (clear of the forefoot)
        tip = toe_zone & (x >= x[foot].max() - 0.015)
        yp, zp = y[tip], z[tip]
        neg, pos = yp < y_mid, yp >= y_mid
        h_neg = (zp[neg].max() - zp[neg].min()) if neg.any() else 0.0
        h_pos = (zp[pos].max() - zp[pos].min()) if pos.any() else 0.0
        side = "-y" if h_neg >= h_pos else "+y"
        print(f"  [regions] big-toe side auto: {side} "
              f"(toe height -y {h_neg*1e3:.1f} mm, +y {h_pos*1e3:.1f} mm)")
    else:
        side = big_toe_side
        print(f"  [regions] big-toe side set to {side}")
    w = big_toe_width_frac * (yt.max() - yt.min())
    big = toe_zone & ((y <= yt.min() + w) if side == "-y" else (y >= yt.max() - w))

    rest = foot & ~toe_zone
    xq = np.quantile(x[rest], [1 / 3, 2 / 3])
    up = N[:, 2] >= 0

    region = np.full(len(V), "", dtype="U24")
    region[leg_cut] = "leg_cut"
    fa = ankle_band & (N[:, 0] > front_ankle_nx)
    region[fa] = "front_ankle"
    region[ankle_band & ~fa] = "ankle_other"
    region[big] = "big_toe"
    region[toe_zone & ~big] = "lesser_toes"

    sl = np.select([x <= xq[0], x <= xq[1]], ["heel", "midfoot"], "forefoot")
    surf = np.where(up, "dorsal", "plantar")
    lab = np.char.add(np.char.add(sl, "_"), surf)
    if split_medial_lateral:
        # foot midline: median y in bins along x, interpolated per vertex
        xr = x[rest]
        edges = np.quantile(xr, np.linspace(0, 1, n_midline_bins + 1))
        cen = 0.5 * (edges[:-1] + edges[1:])
        bi = np.clip(np.searchsorted(edges, xr, side="right") - 1, 0, n_midline_bins - 1)
        yr = y[rest]
        med = np.array([np.median(yr[bi == k]) if (bi == k).any() else np.nan
                        for k in range(n_midline_bins)])
        ok = ~np.isnan(med)
        y_mid_x = np.interp(x, cen[ok], med[ok])
        on_big_side = (y < y_mid_x) if side == "-y" else (y >= y_mid_x)
        lab = np.char.add(np.char.add(lab, "_"),
                          np.where(on_big_side, "medial", "lateral"))
    region[rest] = lab[rest]

    assert (region != "").all(), "some vertices were not assigned a region"
    print(f"  [regions] ankle band starts {(z_ank - z.min())*1e3:.0f} mm above "
          f"the sole; leg cut at {(z_cut - z.min())*1e3:.0f} mm; toe zone "
          f"{toe_length_m*1e3:.0f} mm")
    return region, leg_cut


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def render_heatmap(V, F, scalar_mm, out_png, vmax, title=""):
    fig = plt.figure(figsize=(13, 3.5))
    for i, (elev, azim, name) in enumerate(
            [(20, 30, "iso"), (0, 0, "front"),
             (90, 0, "top (dorsum)"), (0, 90, "lateral")]):
        ax = fig.add_subplot(1, 4, i + 1, projection="3d")
        poly = ax.plot_trisurf(V[:, 0], V[:, 1], V[:, 2], triangles=F,
                               linewidth=0, antialiased=False, shade=False)
        colors = plt.get_cmap("RdBu_r")((scalar_mm + vmax) / (2 * vmax))
        poly.set_facecolors(colors[F].mean(axis=1))
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect([1, 1, 1])
        ax.set_axis_off()
        ax.set_title(name, fontsize=9)
    cax = fig.add_axes([0.92, 0.2, 0.01, 0.6])
    norm = plt.Normalize(-vmax, vmax)
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
    sm.set_array([])
    plt.colorbar(sm, cax=cax, label="signed disp (mm, outward = +)")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


def render_region_bars(region_stats, out_png, title=""):
    rows = [(r, region_stats[r]) for r in REGION_NAMES
            if r in region_stats and "mean_signed_mm" in region_stats[r]]
    if not rows:
        return
    xs = [r[0] for r in rows]
    ys_mean = np.array([r[1]["mean_signed_mm"] for r in rows])
    ys_p5 = np.array([r[1]["p5_signed_mm"] for r in rows])
    ys_p95 = np.array([r[1]["p95_signed_mm"] for r in rows])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["tab:blue" if y >= 0 else "tab:orange" for y in ys_mean]
    ax.bar(xs, ys_mean, color=colors, label="mean signed")
    # clip: with skewed data the mean can fall outside [p5, p95]
    ax.errorbar(xs, ys_mean, yerr=[np.clip(ys_mean - ys_p5, 0, None),
                                   np.clip(ys_p95 - ys_mean, 0, None)],
                fmt="none", ecolor="black", capsize=4, label="p5–p95")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("signed displacement (mm, outward = +)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    plt.xticks(rotation=15)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", default=DEFAULT_D3,
                    help=f"Baseline (reference) mesh, default: D3 healthy")
    ap.add_argument("--followup", default=DEFAULT_D0,
                    help=f"Follow-up mesh, default: D0 edema (outward +)")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help=f"Output dir, default: {DEFAULT_OUT}")
    ap.add_argument("--K", type=int, default=50,
                    help="Number of spectral modes per region (default 50)")
    ap.add_argument("--amp_mm_proto", type=float, default=4.0)
    ap.add_argument("--sigma_m_proto", type=float, default=0.025)
    ap.add_argument("--leg_cut_z_quantile", type=float, default=0.95,
                    help="Vertices with z above this quantile are flagged as "
                         "the leg-cut zone and reported separately.")
    ap.add_argument("--ankle_height_frac", type=float, default=0.45,
                    help="Ankle band starts this fraction of the way from the "
                         "sole to the leg cut (default 0.45).")
    ap.add_argument("--toe_length_mm", type=float, default=45.0,
                    help="Toe zone = this many mm back from the toe tip.")
    ap.add_argument("--big_toe_width_frac", type=float, default=0.38,
                    help="Big toe = this fraction of the toe-zone width.")
    ap.add_argument("--big_toe_side", choices=["auto", "+y", "-y"],
                    default="auto")
    ap.add_argument("--no_medial_lateral", action="store_true",
                    help="Do not split heel/midfoot/forefoot into medial and "
                         "lateral halves.")
    ap.add_argument("--front_ankle_nx", type=float, default=0.3,
                    help="Front ankle = ankle-band vertices whose normal "
                         "x-component exceeds this (faces forward).")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    # [1/7] Load
    print(f"[1/7] Load")
    m_b = trimesh.load(args.baseline, process=False)
    m_f = trimesh.load(args.followup, process=False)
    V_b_raw = np.asarray(m_b.vertices)
    V_f_raw = np.asarray(m_f.vertices)
    F = np.asarray(m_b.faces)
    if V_b_raw.shape != V_f_raw.shape:
        sys.exit(f"vertex counts differ: baseline {len(V_b_raw)} vs "
                 f"follow-up {len(V_f_raw)}")
    if not np.array_equal(F, np.asarray(m_f.faces)):
        print("[warn] baseline and follow-up have different faces; using baseline's")
    print(f"  V={len(V_b_raw)}  F={len(F)}")
    try:
        print(f"  baseline  volume={float(m_b.volume) * 1e6:.1f} cm^3  "
              f"watertight={m_b.is_watertight}")
        print(f"  follow-up volume={float(m_f.volume) * 1e6:.1f} cm^3  "
              f"watertight={m_f.is_watertight}")
    except Exception:
        pass
    pre_rms_mm = float(np.sqrt(((V_f_raw - V_b_raw) ** 2).sum(1).mean())) * 1e3
    pre_max_mm = float(np.linalg.norm(V_f_raw - V_b_raw, axis=1).max()) * 1e3
    print(f"  pre-align RMS {pre_rms_mm:.2f} mm, max {pre_max_mm:.2f} mm")

    # [2/7] Kabsch rigid alignment on all vertices
    print(f"[2/7] Kabsch rigid alignment")
    R_k, t_k, resid_k = kabsch(V_f_raw, V_b_raw)
    V_f_aligned = V_f_raw @ R_k.T + t_k
    post_rms_mm = float(np.sqrt((resid_k ** 2).mean())) * 1e3
    print(f"  post-align RMS {post_rms_mm:.2f} mm, max {resid_k.max() * 1e3:.2f} mm")

    # [3/7] Canonicalise: centre only, keep raw axes (x=length, y=width, z=height).
    # PCA was dropped: it ranked height above width (the leg adds vertical
    # spread) and left the heel-to-toe sign arbitrary, which mislabelled regions.
    print(f"[3/7] Canonicalisation (centre + heel-to-toe sign)")
    c = V_b_raw.mean(axis=0)
    V_b_c = V_b_raw - c
    V_f_c = V_f_aligned - c
    # The leg rises above the heel, so put the heel at -x and the toes at +x
    top = V_b_c[:, 2] >= np.quantile(V_b_c[:, 2], 0.90)
    if V_b_c[top, 0].mean() > 0:
        rot = np.diag([-1.0, -1.0, 1.0])   # 180 deg about z: no mirroring
        V_b_c = V_b_c @ rot
        V_f_c = V_f_c @ rot
        print("  rotated 180 deg about z so toes point to +x")
    m_b_canon = m_b.copy(); m_b_canon.vertices = V_b_c
    N_b_c = np.asarray(m_b_canon.vertex_normals)
    print(f"  canonical bbox  x[{V_b_c[:,0].min():+.3f}, {V_b_c[:,0].max():+.3f}]  "
          f"y[{V_b_c[:,1].min():+.3f}, {V_b_c[:,1].max():+.3f}]  "
          f"z[{V_b_c[:,2].min():+.3f}, {V_b_c[:,2].max():+.3f}]")

    # [4/7] Per-vertex signed displacement
    print(f"[4/7] Signed displacement (vertex-wise correspondence)")
    Delta = V_f_c - V_b_c                     # (V, 3) metres
    signed = np.einsum("ij,ij->i", Delta, N_b_c)
    signed_mm = signed * 1e3
    mag_mm = np.linalg.norm(Delta, axis=1) * 1e3
    print(f"  signed disp  mean={signed_mm.mean():+.3f}  "
          f"median={np.median(signed_mm):+.3f}  "
          f"p5={np.quantile(signed_mm, 0.05):+.3f}  "
          f"p95={np.quantile(signed_mm, 0.95):+.3f}  "
          f"range [{signed_mm.min():+.3f}, {signed_mm.max():+.3f}] mm")
    print(f"  |disp|  mean={mag_mm.mean():.3f}  max={mag_mm.max():.3f} "
          f"at vertex {int(np.argmax(mag_mm))}")

    # [5/7] Regions
    print(f"[5/7] Regions (geometric)")
    region, leg_cut_mask = define_regions(
        V_b_c, N_b_c, args.leg_cut_z_quantile, args.ankle_height_frac,
        args.toe_length_mm / 1000, args.big_toe_width_frac,
        args.big_toe_side, args.front_ankle_nx,
        split_medial_lateral=not args.no_medial_lateral)
    print(f"  leg-cut zone (top {int((1-args.leg_cut_z_quantile)*100)}% z): "
          f"{int(leg_cut_mask.sum())} vertices (reported separately below)")

    region_stats = {}
    for r in REGION_NAMES:
        vidx = np.where(region == r)[0]
        if len(vidx) == 0:
            continue                      # region not used in this mode
        if len(vidx) < 50:
            region_stats[r] = {"n_vertices": int(len(vidx)), "note": "too small"}
            continue
        s = signed_mm[vidx]
        region_stats[r] = {
            "n_vertices": int(len(vidx)),
            "mean_signed_mm": float(s.mean()),
            "median_signed_mm": float(np.median(s)),
            "p5_signed_mm": float(np.quantile(s, 0.05)),
            "p95_signed_mm": float(np.quantile(s, 0.95)),
            "rms_signed_mm": float(np.sqrt((s ** 2).mean())),
            "min_signed_mm": float(s.min()),
            "max_signed_mm": float(s.max()),
            "mean_abs_signed_mm": float(np.mean(np.abs(s))),
        }
    for r in REGION_NAMES:
        st = region_stats.get(r, {})
        if not st:
            continue
        if "mean_signed_mm" not in st:
            print(f"  {r:9s}: {st.get('note', 'n/a')}  (n={st.get('n_vertices', 0)})")
            continue
        print(f"  {r:9s}: n={st['n_vertices']:>6}  "
              f"mean={st['mean_signed_mm']:+6.2f}  "
              f"p5={st['p5_signed_mm']:+6.2f}  p95={st['p95_signed_mm']:+6.2f}  "
              f"rms={st['rms_signed_mm']:5.2f}  mm")
    # Leg-cut zone (not part of clinical anatomy)
    if leg_cut_mask.sum() >= 50:
        s_leg = signed_mm[leg_cut_mask]
        region_stats["_leg_cut_zone"] = {
            "n_vertices": int(leg_cut_mask.sum()),
            "mean_signed_mm": float(s_leg.mean()),
            "rms_signed_mm": float(np.sqrt((s_leg ** 2).mean())),
            "max_abs_signed_mm": float(np.abs(s_leg).max()),
            "note": "high-z vertices possibly affected by leg-cut height difference",
        }
        print(f"  _leg_cut_zone: n={int(leg_cut_mask.sum()):>6}  "
              f"mean={float(s_leg.mean()):+6.2f}  "
              f"max|.|={float(np.abs(s_leg).max()):6.2f}  mm  "
              f"[flagged artifact]")

    # [6/7] Spectral latent per region
    print(f"[6/7] Spectral latent per sizeable region")
    proto_mm = canonical_prototype_field(V_b_c, N_b_c,
                                         amp_mm=args.amp_mm_proto,
                                         sigma_m=args.sigma_m_proto)
    latent_stats = {}
    for r in REGION_NAMES:
        vidx = np.where(region == r)[0]
        if len(vidx) < 100:
            continue
        Vsub, Fsub = sub_mesh(V_b_c, F, vidx)
        K_use = min(args.K, max(5, len(vidx) // 30))
        L = mesh_graph_laplacian(Vsub, Fsub)
        Phi = spectral_basis(L, K_use)
        Delta_region_mm = (signed_mm[:, None] * N_b_c)[vidx]
        alpha = Phi.T @ Delta_region_mm
        alpha_star = Phi.T @ proto_mm[vidx]
        p_norm = float(np.linalg.norm(alpha_star))
        if p_norm > 1e-9:
            s_proto = float(np.sum(alpha * (alpha_star / p_norm)))
        else:
            s_proto = 0.0
        latent_stats[r] = {
            "K": int(K_use),
            "latent_norm_mm": float(np.linalg.norm(alpha)),
            "prototype_score_mm": s_proto,
            "prototype_defined": p_norm > 1e-9,
        }
        print(f"  {r:9s}: K={K_use:>3}  ||alpha||={latent_stats[r]['latent_norm_mm']:7.2f}  "
              f"proto_score={s_proto:+7.2f}  (proto_defined={p_norm > 1e-9})")

    # [7/7] Save
    print(f"[7/7] Save outputs to {args.out}")
    results = {
        "baseline_path": args.baseline,
        "followup_path": args.followup,
        "n_vertices": int(len(V_b_c)),
        "n_faces": int(len(F)),
        "baseline_volume_cm3": float(getattr(m_b, 'volume', 0.0)) * 1e6,
        "followup_volume_cm3": float(getattr(m_f, 'volume', 0.0)) * 1e6,
        "pre_align_rms_mm": pre_rms_mm,
        "pre_align_max_mm": pre_max_mm,
        "post_align_rms_mm": post_rms_mm,
        "signed_disp_global": {
            "mean_mm": float(signed_mm.mean()),
            "median_mm": float(np.median(signed_mm)),
            "rms_mm": float(np.sqrt((signed_mm ** 2).mean())),
            "p5_mm": float(np.quantile(signed_mm, 0.05)),
            "p95_mm": float(np.quantile(signed_mm, 0.95)),
            "min_mm": float(signed_mm.min()),
            "max_mm": float(signed_mm.max()),
            "argmax_abs_vertex": int(np.argmax(np.abs(signed_mm))),
        },
        "region_stats": region_stats,
        "latent_stats": latent_stats,
        "params": {
            "K": args.K,
            "prototype_amp_mm": args.amp_mm_proto,
            "prototype_sigma_m": args.sigma_m_proto,
            "leg_cut_z_quantile": args.leg_cut_z_quantile,
            "ankle_height_frac": args.ankle_height_frac,
            "toe_length_mm": args.toe_length_mm,
            "big_toe_width_frac": args.big_toe_width_frac,
            "big_toe_side": args.big_toe_side,
            "front_ankle_nx": args.front_ankle_nx,
            "split_medial_lateral": not args.no_medial_lateral,
        },
    }
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    vmax = max(1.0, float(np.quantile(np.abs(signed_mm), 0.99)))
    render_heatmap(V_b_c, F, signed_mm,
                   os.path.join(args.out, "signed_disp_heatmap.png"),
                   vmax=vmax,
                   title=f"62L signed displacement  follow-up vs baseline  "
                   f"|disp| p99={vmax:.2f} mm")
    render_region_bars(region_stats,
                       os.path.join(args.out, "region_bars.png"),
                       title=f"62L per-region signed displacement\n"
                       f"baseline={os.path.basename(args.baseline)}, "
                       f"follow-up={os.path.basename(args.followup)}")

    np.save(os.path.join(args.out, "signed_disp_mm.npy"), signed_mm)
    np.save(os.path.join(args.out, "disp_vector_mm.npy"), Delta * 1e3)
    np.save(os.path.join(args.out, "region_labels.npy"), region)
    np.save(os.path.join(args.out, "baseline_vertices_canonical.npy"), V_b_c)
    np.save(os.path.join(args.out, "followup_vertices_canonical.npy"), V_f_c)
    np.save(os.path.join(args.out, "faces.npy"), F)

    # Export processed meshes as PLY for external viewers
    m_b_canon.export(os.path.join(args.out, "baseline_canonical.ply"))
    m_f_canon = m_f.copy(); m_f_canon.vertices = V_f_c
    m_f_canon.export(os.path.join(args.out, "followup_canonical.ply"))

    print(f"\nDone in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
