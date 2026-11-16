import os, json, numpy as np, pyvista as pv
pv.set_jupyter_backend("client")   # renders in your browser, no server display needed

out = f"/node/scratch/{os.environ['USER']}/edema_track_out"
V0 = np.load(f"{out}/baseline_vertices_canonical.npy")
V1 = np.load(f"{out}/followup_vertices_canonical.npy")
F = np.load(f"{out}/faces.npy")
s = np.load(f"{out}/signed_disp_mm.npy")
reg = np.load(f"{out}/region_labels.npy")
res = json.load(open(f"{out}/results.json"))

faces_pv = np.hstack([np.full((len(F), 1), 3), F]).ravel()
base = pv.PolyData(V0, faces_pv)
fol = pv.PolyData(V1, faces_pv)
print(len(V0), s.min(), s.max())

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VIEWS = [(20, 30, "iso"), (0, 0, "front"), (90, 0, "top"), (0, 90, "lateral")]

def render(V, F, colors, path, title="", cbar=None, legend=None):
    fig = plt.figure(figsize=(14, 3.8))
    for i, (elev, azim, name) in enumerate(VIEWS):
        ax = fig.add_subplot(1, 4, i + 1, projection="3d")
        poly = ax.plot_trisurf(V[:, 0], V[:, 1], V[:, 2], triangles=F,
                               linewidth=0, antialiased=False, shade=False)
        poly.set_facecolors(colors[F].mean(axis=1))
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect(np.ptp(V, axis=0))
        ax.set_axis_off(); ax.set_title(name, fontsize=9)
    if cbar is not None:
        cax = fig.add_axes([0.92, 0.2, 0.01, 0.6])
        fig.colorbar(cbar, cax=cax)
    if legend is not None:
        fig.legend(handles=legend, loc="lower center", ncol=len(legend),
                   fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=10)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)

vmax = float(np.quantile(np.abs(s), 0.99))
norm = plt.Normalize(-vmax, vmax)
cmap = plt.get_cmap("RdBu_r")
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
render(V0, F, cmap(norm(s)), f"{out}/viz_signed_disp.png",
       title=f"signed displacement D0 vs D3 (mm), red = outward", cbar=sm)

from matplotlib.patches import Patch
names, ids = np.unique(reg, return_inverse=True)
tab = plt.get_cmap("tab10")
handles = [Patch(color=tab(i % 10), label=str(n)) for i, n in enumerate(names)]
render(V0, F, tab(ids % 10), f"{out}/viz_regions.png",
       title="region labels", legend=handles)
for n, c in zip(names, np.bincount(ids)):
    print(f"{str(n):10s} {int(c)}")

fig, ax = plt.subplots(figsize=(8, 4))
ax.boxplot([s[reg == r] for r in names], tick_labels=[str(n) for n in names],
           showfliers=False)
ax.axhline(0, color="k", lw=0.5)
ax.set_ylabel("signed displacement (mm)")
plt.xticks(rotation=20)
fig.tight_layout()
fig.savefig(f"{out}/viz_region_box.png", dpi=140)
plt.close(fig)
print("saved", f"{out}/viz_region_box.png")

import sys, trimesh
sys.path.insert(0, f"/node/scratch/{os.environ['USER']}")
from edema_track import sub_mesh, mesh_graph_laplacian, spectral_basis, canonical_prototype_field

N0 = np.asarray(trimesh.Trimesh(V0, F, process=False).vertex_normals)
proto_s = np.einsum("ij,ij->i", canonical_prototype_field(V0, N0, amp_mm=4.0, sigma_m=0.025), N0)

regions = [str(r) for r in names if (reg == r).sum() >= 100]
fig, axes = plt.subplots(len(regions), 1, figsize=(9, 2.2 * len(regions)), sharex=True)
for ax, r in zip(np.atleast_1d(axes), regions):
    vidx = np.where(reg == r)[0]
    Vs, Fs = sub_mesh(V0, F, vidx)
    K = min(50, max(5, len(vidx) // 30))
    Phi = spectral_basis(mesh_graph_laplacian(Vs, Fs), K)
    Phi = Phi * np.sign(Phi[np.abs(Phi).argmax(axis=0), np.arange(K)])   # fix sign convention
    a = Phi.T @ s[vidx] / np.sqrt(len(vidx))
    a_p = Phi.T @ proto_s[vidx] / np.sqrt(len(vidx))
    ax.bar(np.arange(K), a, color=np.where(a >= 0, "tab:red", "tab:blue"), label="observed a_k")
    ax.plot(np.arange(K), a_p, "k.-", ms=3, lw=0.8, label="prototype")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel(r, rotation=0, ha="right", fontsize=9)
np.atleast_1d(axes)[0].legend(fontsize=8)
np.atleast_1d(axes)[-1].set_xlabel("spectral mode k (mode 0 = regional mean, mm)")
fig.tight_layout()
fig.savefig(f"{out}/viz_alpha_spectra_signed.png", dpi=140)
plt.close(fig)
print("saved", f"{out}/viz_alpha_spectra_signed.png")

rows = []
for r in regions:
    vidx = np.where(reg == r)[0]
    Vs, Fs = sub_mesh(V0, F, vidx)
    K = min(50, max(5, len(vidx) // 30))
    L = mesh_graph_laplacian(Vs, Fs)
    Phi = spectral_basis(L, K)
    lam = np.einsum("ik,ik->k", Phi, L @ Phi)       # eigenvalues via Rayleigh quotient
    a = Phi.T @ s[vidx]                              # signed scalar coefficients
    lam_n = lam / lam[-1]                            # normalise so t is comparable across regions
    out_r = {"region": r, "n": len(vidx), "K": K,
             "mean_mm": float(s[vidx].mean())}
    for name, w in [("equal", np.ones(K)),
                    ("heat_t2", np.exp(-2 * lam_n)),
                    ("heat_t5", np.exp(-5 * lam_n))]:
        s_w = Phi @ (w * a)
        out_r[f"{name}_mean"] = float(s_w.mean())
        out_r[f"{name}_p95"] = float(np.quantile(s_w, 0.95))
    rows.append(out_r)

import pandas as pd
df = pd.DataFrame(rows).set_index("region").round(2)
df.to_csv(f"{out}/spectral_lowpass_summary.csv")
df