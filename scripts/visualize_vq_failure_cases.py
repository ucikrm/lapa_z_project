from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


parser = argparse.ArgumentParser()
parser.add_argument("--rank_csv", type=str, required=True)
parser.add_argument("--vq_dir", type=str, required=True)
parser.add_argument("--cont_dir", type=str, required=True)
parser.add_argument("--out_dir", type=str, required=True)
parser.add_argument("--top_k", type=int, default=5)
parser.add_argument("--project_root", type=str, default=".")
args = parser.parse_args()

root = Path(args.project_root).resolve()
rank_df = pd.read_csv(args.rank_csv).head(args.top_k)

vq_dir = Path(args.vq_dir)
cont_dir = Path(args.cont_dir)
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

meta = pd.read_csv(vq_dir / "metadata.csv")
Z_vq = np.load(vq_dir / "z_quantized.npy")
codes = np.load(vq_dir / "indices.npy")
Z_cont = np.load(cont_dir / "z_continuous.npy")

meta["traj"] = meta["x1_path"].apply(lambda p: Path(str(p)).parent.name)

def resolve_path(p):
    p = Path(str(p))
    candidates = [
        p,
        root / p,
        root / "scripts" / p,
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Could not resolve {p}")

for _, row in rank_df.iterrows():
    traj = row["traj"]
    group = meta[meta["traj"] == traj].copy()
    idxs = group.index.to_numpy()

    if len(idxs) < 2:
        continue

    z_vq = Z_vq[idxs]
    z_cont = Z_cont[idxs]
    c = codes[idxs]

    # collect up to 8 frames from x1 plus final x2
    frame_paths = list(group["x1_path"].values)
    frame_paths.append(group["x2_path"].values[-1])
    sample_ids = np.linspace(0, len(frame_paths) - 1, min(8, len(frame_paths)), dtype=int)
    chosen_paths = [resolve_path(frame_paths[i]) for i in sample_ids]

    imgs = [Image.open(p).convert("RGB").resize((128, 128)) for p in chosen_paths]

    fig = plt.figure(figsize=(14, 7))
    fig.suptitle(
        f"{traj} | {row['label']}\n"
        f"VQ codes={int(row['vq_used_codes'])}, top_frac={row['vq_top_code_fraction']:.2f}, "
        f"rank VQ={row['vq_effective_rank']:.2f}, rank Continuous={row['continuous_effective_rank']:.2f}",
        fontsize=11,
    )

    for i, img in enumerate(imgs):
        ax = plt.subplot(3, max(8, len(imgs)), i + 1)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"t{i}", fontsize=8)

    # VQ code sequence
    ax = plt.subplot(3, 2, 3)
    ax.plot(np.arange(len(c)), c, marker="o")
    ax.set_title("VQ code over time")
    ax.set_xlabel("transition t")
    ax.set_ylabel("code id")

    # Continuous PCA trajectory
    ax = plt.subplot(3, 2, 4)
    if len(z_cont) >= 2:
        z2 = PCA(n_components=2).fit_transform(z_cont)
        ax.plot(z2[:, 0], z2[:, 1], marker="o")
        for i in range(len(z2)):
            ax.text(z2[i, 0], z2[i, 1], str(i), fontsize=6)
    ax.set_title("Continuous Z trajectory, PCA")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    # VQ PCA trajectory
    ax = plt.subplot(3, 2, 5)
    if len(z_vq) >= 2:
        z2v = PCA(n_components=2).fit_transform(z_vq)
        ax.plot(z2v[:, 0], z2v[:, 1], marker="o")
        for i in range(len(z2v)):
            ax.text(z2v[i, 0], z2v[i, 1], str(i), fontsize=6)
    ax.set_title("VQ Z trajectory, PCA")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    # Continuous step distance
    ax = plt.subplot(3, 2, 6)
    if len(z_cont) >= 2:
        step_dist = np.linalg.norm(np.diff(z_cont, axis=0), axis=1)
        ax.plot(step_dist, marker="o")
    ax.set_title("Continuous Z step distance")
    ax.set_xlabel("transition t")
    ax.set_ylabel("||Z[t+1]-Z[t]||")

    plt.tight_layout()
    out_path = out_dir / f"{traj}_failure_panel.png"
    plt.savefig(out_path, dpi=200)
    plt.close()

    print("Saved:", out_path)
