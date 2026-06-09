from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


parser = argparse.ArgumentParser()
parser.add_argument("--vq_dir", type=str, required=True)
parser.add_argument("--naive_dir", type=str, required=True)
parser.add_argument("--cont_dir", type=str, required=True)
parser.add_argument("--fig_dir", type=str, required=True)
args = parser.parse_args()

vq_dir = Path(args.vq_dir)
naive_dir = Path(args.naive_dir)
cont_dir = Path(args.cont_dir)
fig_dir = Path(args.fig_dir)
fig_dir.mkdir(parents=True, exist_ok=True)

Z_vq = np.load(vq_dir / "z_quantized.npy")
codes = np.load(vq_dir / "indices.npy")
Z_naive = np.load(naive_dir / "z_naive_continuous.npy")
Z_cont = np.load(cont_dir / "z_continuous.npy")

print("VQ Z:", Z_vq.shape)
print("Naive continuous Z:", Z_naive.shape)
print("Anti-collapse continuous Z:", Z_cont.shape)

def effective_rank(Z):
    Z = Z - Z.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Z, compute_uv=False)
    if s.sum() <= 1e-12:
        return 0.0
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p + 1e-12)).sum()))

def summarize(name, Z):
    std = Z.std(axis=0)
    return {
        "method": name,
        "mean_abs": float(np.abs(Z).mean()),
        "std_mean": float(std.mean()),
        "std_min": float(std.min()),
        "std_max": float(std.max()),
        "effective_rank": effective_rank(Z[:min(len(Z), 5000)]),
    }

rows = [
    summarize("VQ/codebook", Z_vq),
    summarize("Naive continuous", Z_naive),
    summarize("Continuous + anti-collapse", Z_cont),
]

summary = pd.DataFrame(rows)
summary.to_csv(fig_dir / "threeway_z_summary.csv", index=False)

print()
print(summary.to_string(index=False))

unique, counts = np.unique(codes, return_counts=True)
usage = pd.DataFrame({"code": unique, "count": counts})
usage["fraction"] = usage["count"] / usage["count"].sum()
usage.to_csv(fig_dir / "vq_code_usage.csv", index=False)

entropy = float(-(usage["fraction"] * np.log(usage["fraction"] + 1e-12)).sum())

print()
print("VQ used codes:", len(unique))
print("VQ entropy:", entropy)
print("VQ top codes:")
print(usage.sort_values("count", ascending=False).head(10))

def pca_plot(Z, title, path, color=None):
    n = min(len(Z), 10000)
    Zs = Z[:n]
    Z2 = PCA(n_components=2).fit_transform(Zs)

    plt.figure(figsize=(6, 6))
    if color is None:
        plt.scatter(Z2[:, 0], Z2[:, 1], s=6)
    else:
        plt.scatter(Z2[:, 0], Z2[:, 1], s=6, c=color[:n], cmap="tab20")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

pca_plot(Z_vq, "VQ/codebook Z PCA", fig_dir / "vq_z_pca.png", color=codes)
pca_plot(Z_naive, "Naive continuous Z PCA", fig_dir / "naive_continuous_z_pca.png")
pca_plot(Z_cont, "Continuous + anti-collapse Z PCA", fig_dir / "anticollapse_continuous_z_pca.png")

# Bar chart: effective rank.
plt.figure(figsize=(8, 4))
plt.bar(summary["method"], summary["effective_rank"])
plt.ylabel("Effective rank")
plt.xticks(rotation=20, ha="right")
plt.title("Latent Z effective rank comparison")
plt.tight_layout()
plt.savefig(fig_dir / "threeway_effective_rank.png", dpi=200)
plt.close()

# Bar chart: std mean.
plt.figure(figsize=(8, 4))
plt.bar(summary["method"], summary["std_mean"])
plt.ylabel("Mean per-dimension std")
plt.xticks(rotation=20, ha="right")
plt.title("Latent Z diversity comparison")
plt.tight_layout()
plt.savefig(fig_dir / "threeway_std_mean.png", dpi=200)
plt.close()

print()
print("Saved figures to:", fig_dir)
