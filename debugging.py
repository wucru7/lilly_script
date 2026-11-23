x, z = V0[:, 0], V0[:, 2]
nx = np.asarray(trimesh.Trimesh(V0, F, process=False).vertex_normals)[:, 0]
z0 = z.min(); z_cut = np.quantile(z, 0.95)
front = nx > 0.3
bins = np.linspace(z0, z.max(), 41); c = 0.5 * (bins[:-1] + bins[1:])
b = np.digitize(z[front], bins) - 1
m = [s[front][b == k].mean() if (b == k).sum() > 20 else np.nan for k in range(40)]

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot((c - z0) * 1000, m, "o-")
ax.axhline(0, color="k", lw=0.5)
z_ank = z0 + 0.45 * (z_cut - z0)
ax.axvline((z_ank - z0) * 1000, color="tab:green", ls="--", label="front_ankle starts (0.45)")
ax.axvline((z_cut - z0) * 1000, color="tab:red", ls="--", label="leg_cut starts (q 0.95)")
ax.set_xlabel("height above sole (mm)"); ax.set_ylabel("mean signed disp, front-facing (mm)")
ax.legend(); fig.tight_layout(); fig.savefig(f"{run}/viz_front_profile.png", dpi=140); plt.close(fig)
for k in range(40):
    if not np.isnan(m[k]):
        print(f"{(c[k]-z0)*1000:6.1f} mm  frac={(c[k]-z0)/(z_cut-z0):.2f}  mean={m[k]:+.2f}")