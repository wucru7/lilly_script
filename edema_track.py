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
  5. Geometric 7-region partition (heel / midfoot / forefoot / toes / dorsum /
     sole / ankle) + a "leg-cut zone" bucket that reports the top-z vertices
     separately so leg-cut differences don't contaminate the clinical regions.
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
REGION_NAMES = ["heel", "midfoot", "forefoot", "toes",
                "dorsum", "sole", "ankle"]


def define_regions(V, N, leg_cut_z_quantile=0.95):
    """Partition baseline vertices into 7 regions by position + normal.

    Also returns a separate leg-cut mask (top z quantile) — the leg-cut
    region is left in its anatomical bucket but reported separately so
    differing leg-cut heights between visits don't masquerade as swelling.
    """
    xq = np.quantile(V[:, 0], [0.25, 0.50, 0.75])
    zq = np.quantile(V[:, 2], [0.50, leg_cut_z_quantile])
    region = np.full(len(V), "midfoot", dtype="U10")
    region[V[:, 0] <= xq[0]] = "heel"
    region[(V[:, 0] > xq[0]) & (V[:, 0] <= xq[1])] = "midfoot"
    region[(V[:, 0] > xq[1]) & (V[:, 0] <= xq[2])] = "forefoot"
    region[V[:, 0] > xq[2]] = "toes"
    region[(N[:, 2] > 0.3) & (V[:, 2] >= zq[0])] = "dorsum"
    region[N[:, 2] < -0.3] = "sole"
    region[V[:, 2] >= zq[1]] = "ankle"
    leg_cut_mask = V[:, 2] >= zq[1]
    return region, leg_cut_mask


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
    region, leg_cut_mask = define_regions(V_b_c, N_b_c,
                                          args.leg_cut_z_quantile)
    print(f"  leg-cut zone (top {int((1-args.leg_cut_z_quantile)*100)}% z): "
          f"{int(leg_cut_mask.sum())} vertices (reported separately below)")

    region_stats = {}
    for r in REGION_NAMES:
        vidx = np.where(region == r)[0]
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
