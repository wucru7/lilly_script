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