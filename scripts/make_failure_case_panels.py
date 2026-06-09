from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as T

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

sys.path.append(str(Path(__file__).resolve().parent))
from phase1_models import VQPhase1Model, ContinuousPhase1Model


parser = argparse.ArgumentParser()
parser.add_argument("--pairwise_csv", type=str, required=True)
parser.add_argument("--vq_ckpt", type=str, required=True)
parser.add_argument("--cont_ckpt", type=str, required=True)
parser.add_argument("--vq_z_dir", type=str, required=True)
parser.add_argument("--cont_z_dir", type=str, required=True)
parser.add_argument("--out_dir", type=str, required=True)
parser.add_argument("--top_k", type=int, default=5)
parser.add_argument("--image_size", type=int, default=128)
parser.add_argument("--project_root", type=str, default=".")
args = parser.parse_args()

project_root = Path(args.project_root).resolve()
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = T.Compose([
    T.Resize((args.image_size, args.image_size)),
    T.ToTensor(),
])

def resolve_path(p):
    p = Path(str(p))
    candidates = [p, project_root / p, project_root / "scripts" / p]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Could not resolve path: {p}")

def load_img_tensor(path):
    img = Image.open(resolve_path(path)).convert("RGB")
    return transform(img)

def tensor_to_np(x):
    x = x.detach().cpu().clamp(0, 1)
    return x.permute(1, 2, 0).numpy()

vq_ckpt = torch.load(args.vq_ckpt, map_location="cpu")
cont_ckpt = torch.load(args.cont_ckpt, map_location="cpu")

vq_args = vq_ckpt["args"]
cont_args = cont_ckpt["args"]

vq_model = VQPhase1Model(
    latent_dim=vq_args["latent_dim"],
    num_codes=vq_args["num_codes"],
).to(device)
vq_model.load_state_dict(vq_ckpt["model"])
vq_model.eval()

cont_model = ContinuousPhase1Model(
    latent_dim=cont_args["latent_dim"],
).to(device)
cont_model.load_state_dict(cont_ckpt["model"])
cont_model.eval()

df = pd.read_csv(args.pairwise_csv)
traj_df = df.drop_duplicates("traj").sort_values("failure_score", ascending=False)
top_trajs = traj_df.head(args.top_k)["traj"].tolist()

Z_vq = np.load(Path(args.vq_z_dir) / "z_quantized.npy")
Z_cont = np.load(Path(args.cont_z_dir) / "z_continuous.npy")
codes = np.load(Path(args.vq_z_dir) / "indices.npy")

with torch.no_grad():
    for rank, traj in enumerate(top_trajs, start=1):
        g = df[df["traj"] == traj].copy().sort_values("idx")
        idxs = g["idx"].to_numpy()

        if len(g) < 2:
            continue

        # Choose up to 4 representative transitions with largest continuous advantage.
        g_show = g.sort_values("mse_advantage_cont", ascending=False).head(4)
        label = str(g["label"].iloc[0])

        fig = plt.figure(figsize=(16, 10))
        fig.suptitle(
            f"Rank {rank}: {traj}\n{label}\n"
            f"mean VQ MSE={g['vq_mse'].mean():.5f}, mean Continuous MSE={g['cont_mse'].mean():.5f}, "
            f"advantage={g['mse_advantage_cont'].mean():.5f}\n"
            f"VQ used codes={int(g['traj_vq_used_codes'].iloc[0])}, "
            f"top code frac={g['traj_vq_top_code_fraction'].iloc[0]:.2f}, "
            f"rank VQ={g['traj_vq_effective_rank'].iloc[0]:.2f}, "
            f"rank Continuous={g['traj_cont_effective_rank'].iloc[0]:.2f}",
            fontsize=11,
        )

        # First rows: x1, x2, VQ recon, Continuous recon for selected transitions.
        for col, (_, row) in enumerate(g_show.iterrows()):
            x1 = load_img_tensor(row["x1_path"]).unsqueeze(0).to(device)
            x2 = load_img_tensor(row["x2_path"]).unsqueeze(0).to(device)

            out_vq = vq_model(x1, x2)
            out_cont = cont_model(x1, x2)

            imgs = [
                ("x_t", x1[0]),
                ("x_t+1", x2[0]),
                (f"VQ recon\nMSE={row['vq_mse']:.4f}", out_vq["x2_hat"][0]),
                (f"Cont recon\nMSE={row['cont_mse']:.4f}", out_cont["x2_hat"][0]),
            ]

            for r, (title, img_t) in enumerate(imgs):
                ax = plt.subplot(5, 4, r * 4 + col + 1)
                ax.imshow(tensor_to_np(img_t))
                ax.axis("off")
                ax.set_title(title, fontsize=8)

        # VQ code trajectory.
        ax = plt.subplot(5, 3, 13)
        c = codes[idxs]
        ax.plot(np.arange(len(c)), c, marker="o")
        ax.set_title("VQ code over time")
        ax.set_xlabel("transition")
        ax.set_ylabel("code id")

        # VQ PCA.
        ax = plt.subplot(5, 3, 14)
        z_vq = Z_vq[idxs]
        if len(z_vq) >= 2:
            z2 = PCA(n_components=2).fit_transform(z_vq)
            ax.plot(z2[:, 0], z2[:, 1], marker="o")
        ax.set_title("VQ Z trajectory PCA")

        # Continuous PCA.
        ax = plt.subplot(5, 3, 15)
        z_cont = Z_cont[idxs]
        if len(z_cont) >= 2:
            z2 = PCA(n_components=2).fit_transform(z_cont)
            ax.plot(z2[:, 0], z2[:, 1], marker="o")
        ax.set_title("Continuous Z trajectory PCA")

        plt.tight_layout()
        out_path = out_dir / f"rank_{rank:02d}_{traj}.png"
        plt.savefig(out_path, dpi=200)
        plt.close()
        print("Saved:", out_path)
