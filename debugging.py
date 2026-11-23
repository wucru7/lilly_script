import os, numpy as np
out = f"/node/scratch/{os.environ['USER']}/edema_track_out"
V0 = np.load(f"{out}/baseline_vertices_canonical.npy")
F = np.load(f"{out}/faces.npy")
reg = np.load(f"{out}/region_labels.npy")

import pandas as pd, trimesh
N0 = np.asarray(trimesh.Trimesh(V0, F, process=False).vertex_normals)
x, z, nz = V0[:, 0], V0[:, 2], N0[:, 2]

# rebuild each rule exactly as define_regions does
xq = np.quantile(x, [0.25, 0.50, 0.75])
zq = np.quantile(z, [0.50, 0.95])
slice_ = np.select([x <= xq[0], x <= xq[1], x <= xq[2]],
                   ["heel", "midfoot", "forefoot"], "toes")
rules = {"dorsum": (nz > 0.3) & (z >= zq[0]),
         "sole":   nz < -0.3,
         "ankle":  z >= zq[1]}

# 1) where the final labels came from: rows = length slice, cols = final label
ct = pd.crosstab(pd.Series(slice_, name="length slice"),
                 pd.Series(reg.astype(str), name="final label"))
print(ct, "\n")
print("share of each slice that kept its own label:")
for s_ in ["heel", "midfoot", "forefoot", "toes"]:
    print(f"  {s_:9s} {ct.loc[s_].get(s_, 0) / ct.loc[s_].sum():.0%}")

# 2) overlap between the override rules themselves
print("\nrule overlaps (vertices):")
print("  dorsum & ankle:", int((rules["dorsum"] & rules["ankle"]).sum()),
      "  sole & ankle:", int((rules["sole"] & rules["ankle"]).sum()))

# 3) how far each final region stretches along the foot
print("\nlength-wise extent of final regions (mm along x):")
for r in np.unique(reg):
    xr = x[reg == r] * 1000
    print(f"  {str(r):9s} n={int((reg == r).sum()):>6}  x from {xr.min():+7.1f} to {xr.max():+7.1f}  "
          f"(span {np.ptp(xr):5.1f} mm)")