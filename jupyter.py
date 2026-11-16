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

vmax = np.quantile(np.abs(s), 0.99)
base["signed_mm"] = s
p = pv.Plotter()
p.add_mesh(base, scalars="signed_mm", cmap="RdBu_r", clim=[-vmax, vmax])
p.show()

names, ids = np.unique(reg, return_inverse=True)
base["region"] = ids
p = pv.Plotter()
p.add_mesh(base, scalars="region", cmap="tab10", categories=True)
p.show()
print(dict(zip(names, np.bincount(ids))))

p = pv.Plotter()
p.add_mesh(base, color="lightgray", opacity=0.5, label="D3 healthy")
p.add_mesh(fol, color="tomato", opacity=0.5, label="D0 edema")
p.add_legend()
p.show()

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 4))
order = [r for r in names]
ax.boxplot([s[reg == r] for r in order], labels=order, showfliers=False)
ax.axhline(0, color="k", lw=0.5)
ax.set_ylabel("signed displacement (mm)")
plt.xticks(rotation=20)
plt.show()