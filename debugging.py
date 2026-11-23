import os, sys, glob, getpass, importlib
import numpy as np, pandas as pd, trimesh
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

USER = os.environ.get("USER") or getpass.getuser()
run = sorted(glob.glob(f"/node/scratch/{USER}/runs/*"))[-1]
print("user:", USER, " using:", run)

V0 = np.load(f"{run}/baseline_vertices_canonical.npy")
F = np.load(f"{run}/faces.npy")
s = np.load(f"{run}/signed_disp_mm.npy")
reg = np.load(f"{run}/region_labels.npy").astype(str)

sys.path.insert(0, f"/node/scratch/{USER}")
import edema_track; importlib.reload(edema_track)

KLOW = 5
rows = []
for r in np.unique(reg):
    if r == "leg_cut" or (reg == r).sum() < 100:
        continue
    vidx = np.where(reg == r)[0]
    Vs, Fs = edema_track.sub_mesh(V0, F, vidx)
    K = min(50, max(5, len(vidx) // 30))
    Phi = edema_track.spectral_basis(edema_track.mesh_graph_laplacian(Vs, Fs), K)
    a = Phi.T @ s[vidx]                                   # signed coefficients
    s5 = Phi[:, :KLOW] @ a[:KLOW]                         # sign-invariant low-pass field (mm)
    rows.append({"region": r, "n": len(vidx),
                 "mean_mm": a[0] * Phi[0, 0],             # mode 0 = regional mean
                 "E5_mm": np.linalg.norm(a[:KLOW]) / np.sqrt(len(vidx)),
                 "bump5_p95_mm": np.quantile(s5, 0.95),
                 "frac5": float((a[:KLOW] ** 2).sum() / max((a ** 2).sum(), 1e-12))})

df = pd.DataFrame(rows).set_index("region").round(3).sort_values("bump5_p95_mm", ascending=False)
df.to_csv(f"{run}/score_low5.csv")
print(df)

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(df.index, df["bump5_p95_mm"], color=np.where(df["bump5_p95_mm"] >= 0, "tab:red", "tab:blue"))
ax.axhline(0, color="k", lw=0.5)
ax.set_ylabel("outward bump, first 5 modes, p95 (mm)")
plt.xticks(rotation=35, ha="right")
fig.tight_layout()
fig.savefig(f"{run}/viz_score_low5.png", dpi=140)
plt.close(fig)
print("saved", f"{run}/viz_score_low5.png")