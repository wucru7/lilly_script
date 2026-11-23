import os, glob, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

run = sorted(glob.glob(f"/node/scratch/{os.environ['USER']}/runs/*"))[-1]
print("using", run)
V0 = np.load(f"{run}/baseline_vertices_canonical.npy")
F = np.load(f"{run}/faces.npy")
s = np.load(f"{run}/signed_disp_mm.npy")

vmax = float(np.quantile(np.abs(s), 0.99))
norm = plt.Normalize(-vmax, vmax); cm = plt.get_cmap("RdBu_r")
face_col = cm(norm(s))[F].mean(axis=1)

fig = plt.figure(figsize=(12, 5))
for i, (elev, azim, name) in enumerate([(90, -90, "top (dorsum)"), (-90, -90, "bottom (sole)")]):
    ax = fig.add_subplot(1, 2, i + 1, projection="3d")
    poly = ax.plot_trisurf(V0[:, 0], V0[:, 1], V0[:, 2], triangles=F,
                           linewidth=0, antialiased=False, shade=False)
    poly.set_facecolors(face_col)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect(np.ptp(V0, axis=0)); ax.set_axis_off()
    ax.set_title(name, fontsize=10)
fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cm), cax=fig.add_axes([0.92, 0.2, 0.012, 0.6]),
             label="signed disp (mm), red = outward")
fig.suptitle("signed displacement D0 vs D3, top and bottom", fontsize=11)
fig.savefig(f"{run}/viz_signed_disp_top_bottom.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved", f"{run}/viz_signed_disp_top_bottom.png")