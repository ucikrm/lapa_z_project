from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

root = Path("..")
fig_dir = root / "figures"
fig_dir.mkdir(exist_ok=True)

Z_vq = np.load(root / "z_exports/vq_lapa/z_quantized.npy")
indices = np.load(root / "z_exports/vq_lapa/indices.npy")
Z_cont = np.load(root / "z_exports/continuous_z/z_continuous.npy")

print("VQ Z:", Z_vq.shape)
print("Continuous Z:", Z_cont.shape)

def effective_rank(Z):
    Z = Z - Z.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Z, compute_uv=False)
    p = s / (s.sum() + 1e-12)
    return np.exp(-(p * np.log(p + 1e-12)).sum())

def summarize_z(name, Z):
    std = Z.std(axis=0)
    sample = Z[:min(len(Z), 5000)]
    print()
    print("==", name, "==")
    print("mean abs:", np.abs(Z).mean())
    print("std mean:", std.mean())
    print("std min:", std.min())
    print("std max:", std.max())
    print("effective rank:", effective_rank(sample))

summarize_z("VQ quantized Z", Z_vq)
summarize_z("Continuous Z", Z_cont)

unique, counts = np.unique(indices, return_counts=True)
usage = pd.DataFrame({"code": unique, "count": counts})
usage["fraction"] = usage["count"] / usage["count"].sum()
usage.to_csv(fig_dir / "vq_code_usage.csv", index=False)

entropy = -(usage["fraction"] * np.log(usage["fraction"] + 1e-12)).sum()
print()
print("== VQ codebook usage ==")
print("used codes:", len(unique))
print("entropy:", entropy)
print("top codes:")
print(usage.sort_values("count", ascending=False).head(10))

def pca_plot(Z, title, path, color=None):
    n = min(len(Z), 10000)
    Zs = Z[:n]
    Z2 = PCA(n_components=2).fit_transform(Zs)

    plt.figure(figsize=(6, 6))
    if color is None:
        plt.scatter(Z2[:, 0], Z2[:, 1], s=4)
    else:
        plt.scatter(Z2[:, 0], Z2[:, 1], s=4, c=color[:n], cmap="tab20")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

pca_plot(Z_vq, "VQ Quantized Z PCA colored by code",
         fig_dir / "vq_z_pca.png", color=indices)

pca_plot(Z_cont, "Continuous Z PCA",
         fig_dir / "continuous_z_pca.png")

print()
print("Saved:")
print(fig_dir / "vq_z_pca.png")
print(fig_dir / "continuous_z_pca.png")
print(fig_dir / "vq_code_usage.csv")
