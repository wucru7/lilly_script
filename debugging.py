import os, sys, json, subprocess, importlib, datetime
import numpy as np, pandas as pd, trimesh
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

USER = os.environ["USER"]
EXTRA_ARGS = ["--big_toe_side=-y"]   # add your other settings here, e.g. "--ankle_height_frac", "0.70"
script = f"/node/scratch/{USER}/edema_track.py"
assert 'split = sl != "heel"' in open(script).read(), \
    "edema_track.py on the server is an old version; copy the latest one first"

run = f"/node/scratch/{USER}/runs/{datetime.datetime.now():%Y%m%d_%H%M%S}"
os.makedirs(run, exist_ok=True)

# 1) run the script into the new folder, keep its printout as run_log.txt
p = subprocess.run([sys.executable, script, "--out", run] + EXTRA_ARGS,
                   capture_output=True, text=True)
open(f"{run}/run_log.txt", "w").write(p.stdout + p.stderr)
print(p.stdout[-3000:] if p.returncode == 0 else p.stderr[-3000:])
assert p.returncode == 0, "script failed, see run_log.txt"

# 2) load outputs
V0 = np.load(f"{run}/baseline_vertices_canonical.npy"); F = np.load(f"{run}/faces.npy")
s = np.load(f"{run}/signed_disp_mm.npy"); reg = np.load(f"{run}/region_labels.npy").astype(str)
res = json.load(open(f"{run}/results.json"))
names, ids = np.unique(reg, return_inverse=True)

VIEWS = [(20, 30, "iso"), (0, 0, "front"), (90, -90, "top"),
         (0, 90, "lateral"), (-90, -90, "bottom")]
def render(colors, path, title="", cbar=None, legend=None):
    fig = plt.figure(figsize=(18, 3.8))
    for i, (el, az, nm) in enumerate(VIEWS):
        ax = fig.add_subplot(1, len(VIEWS), i + 1, projection="3d")
        poly = ax.plot_trisurf(V0[:, 0], V0[:, 1], V0[:, 2], triangles=F,
                               linewidth=0, antialiased=False, shade=False)
        poly.set_facecolors(colors[F].mean(axis=1))
        ax.view_init(elev=el, azim=az); ax.set_box_aspect(np.ptp(V0, axis=0))
        ax.set_axis_off(); ax.set_title(nm, fontsize=9)
    if cbar is not None:
        fig.colorbar(cbar, cax=fig.add_axes([0.92, 0.2, 0.01, 0.6]))
    if legend is not None:
        fig.legend(handles=legend, loc="lower center", ncol=min(len(legend), 6),
                   fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.12))
    fig.suptitle(title, fontsize=10)
    fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)

# 3) region map
tab20 = plt.get_cmap("tab20")
render(tab20(ids % 20), f"{run}/viz_regions.png", "region labels",
       legend=[Patch(color=tab20(i % 20), label=n) for i, n in enumerate(names)])

# 4) signed displacement heatmap (now includes the sole)
vmax = float(np.quantile(np.abs(s), 0.99)); norm = plt.Normalize(-vmax, vmax)
cm = plt.get_cmap("RdBu_r")
render(cm(norm(s)), f"{run}/viz_signed_disp.png",
       "signed displacement D0 vs D3 (mm), red = outward",
       cbar=plt.cm.ScalarMappable(norm=norm, cmap=cm))

# 5) per-region box plot (leg_cut shown last as reference)
order = [n for n in names if n != "leg_cut"] + (["leg_cut"] if "leg_cut" in names else [])
fig, ax = plt.subplots(figsize=(12, 4))
ax.boxplot([s[reg == r] for r in order], tick_labels=order, showfliers=False)
ax.axhline(0, color="k", lw=0.5); ax.set_ylabel("signed displacement (mm)")
plt.xticks(rotation=35, ha="right"); fig.tight_layout()
fig.savefig(f"{run}/viz_region_box.png", dpi=140); plt.close(fig)

# 6) signed spectra per region (same functions as the script)
sys.path.insert(0, f"/node/scratch/{USER}")
import edema_track; importlib.reload(edema_track)
N0 = np.asarray(trimesh.Trimesh(V0, F, process=False).vertex_normals)
proto_s = np.einsum("ij,ij->i", edema_track.canonical_prototype_field(V0, N0), N0)
regs = [r for r in order if r != "leg_cut" and (reg == r).sum() >= 100]
fig, axes = plt.subplots(len(regs), 1, figsize=(9, 1.8 * len(regs)), sharex=True)
for ax, r in zip(np.atleast_1d(axes), regs):
    vidx = np.where(reg == r)[0]
    Vs, Fs = edema_track.sub_mesh(V0, F, vidx)
    K = min(50, max(5, len(vidx) // 30))
    Phi = edema_track.spectral_basis(edema_track.mesh_graph_laplacian(Vs, Fs), K)
    Phi = Phi * np.sign(Phi[np.abs(Phi).argmax(axis=0), np.arange(K)])
    a = Phi.T @ s[vidx] / np.sqrt(len(vidx)); a_p = Phi.T @ proto_s[vidx] / np.sqrt(len(vidx))
    ax.bar(np.arange(K), a, color=np.where(a >= 0, "tab:red", "tab:blue"))
    ax.plot(np.arange(K), a_p, "k.-", ms=3, lw=0.8)
    ax.axhline(0, color="k", lw=0.5); ax.set_ylabel(r, rotation=0, ha="right", fontsize=8)
np.atleast_1d(axes)[-1].set_xlabel("spectral mode k (mode 0 = regional mean, mm)")
fig.tight_layout(); fig.savefig(f"{run}/viz_spectra_signed.png", dpi=140); plt.close(fig)

# 7) summary table
rs = {k: v for k, v in res["region_stats"].items() if "mean_signed_mm" in v}
tab = pd.DataFrame(rs).T[["n_vertices", "mean_signed_mm", "median_signed_mm",
                          "p5_signed_mm", "p95_signed_mm", "rms_signed_mm"]].astype(float)
tab = tab.join(pd.DataFrame(res["latent_stats"]).T[["latent_norm_mm", "prototype_score_mm"]].astype(float))
tab = tab.round(2).sort_values("p95_signed_mm", ascending=False)
tab.to_csv(f"{run}/region_summary.csv")
print(tab)
print("\npost-align RMS (mm):", round(res["post_align_rms_mm"], 2))
print("saved everything to", run)