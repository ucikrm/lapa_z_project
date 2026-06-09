from pathlib import Path
import argparse
import sys
import csv

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as T

sys.path.append(str(Path(__file__).resolve().parent))
from phase1_models import VQPhase1Model, ContinuousPhase1Model


parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, required=True)
parser.add_argument("--subset_meta", type=str, required=True)
parser.add_argument("--vq_ckpt", type=str, required=True)
parser.add_argument("--cont_ckpt", type=str, required=True)
parser.add_argument("--vq_z_dir", type=str, required=True)
parser.add_argument("--cont_z_dir", type=str, required=True)
parser.add_argument("--out_csv", type=str, required=True)
parser.add_argument("--image_size", type=int, default=128)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--project_root", type=str, default=".")
args = parser.parse_args()

project_root = Path(args.project_root).resolve()
device = "cuda" if torch.cuda.is_available() else "cpu"

transform = T.Compose([
    T.Resize((args.image_size, args.image_size)),
    T.ToTensor(),
])

def resolve_path(p):
    p = Path(str(p))
    candidates = [
        p,
        project_root / p,
        project_root / "scripts" / p,
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Could not resolve path: {p}")

def load_img(path):
    img = Image.open(path).convert("RGB")
    return transform(img)

def effective_rank(Z):
    if len(Z) < 2:
        return 0.0
    Z = Z - Z.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Z, compute_uv=False)
    if s.sum() <= 1e-12:
        return 0.0
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p + 1e-12)).sum()))

# Load checkpoints.
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

# Load exported metadata and Zs.
meta = pd.read_csv(Path(args.vq_z_dir) / "metadata.csv")
Z_vq = np.load(Path(args.vq_z_dir) / "z_quantized.npy")
Z_cont = np.load(Path(args.cont_z_dir) / "z_continuous.npy")
codes = np.load(Path(args.vq_z_dir) / "indices.npy")

subset_meta = pd.read_csv(args.subset_meta)
label_map = {}
for _, row in subset_meta.iterrows():
    traj_name = f"traj_{int(row['traj_idx']):06d}"
    label_map[traj_name] = str(row.get("label", ""))

meta["traj"] = meta["x1_path"].apply(lambda p: Path(str(p)).parent.name)
meta["label"] = meta["traj"].map(label_map).fillna("")

rows = []

with torch.no_grad():
    for start in range(0, len(meta), args.batch_size):
        end = min(start + args.batch_size, len(meta))
        batch_meta = meta.iloc[start:end]

        x1_list = []
        x2_list = []

        for _, row in batch_meta.iterrows():
            x1_list.append(load_img(resolve_path(row["x1_path"])))
            x2_list.append(load_img(resolve_path(row["x2_path"])))

        x1 = torch.stack(x1_list).to(device)
        x2 = torch.stack(x2_list).to(device)

        out_vq = vq_model(x1, x2)
        out_cont = cont_model(x1, x2)

        # Per-example MSE.
        vq_mse = F.mse_loss(out_vq["x2_hat"], x2, reduction="none")
        vq_mse = vq_mse.flatten(1).mean(dim=1).cpu().numpy()

        cont_mse = F.mse_loss(out_cont["x2_hat"], x2, reduction="none")
        cont_mse = cont_mse.flatten(1).mean(dim=1).cpu().numpy()

        for j, (_, row) in enumerate(batch_meta.iterrows()):
            idx = start + j
            rows.append({
                "idx": idx,
                "traj": row["traj"],
                "label": row["label"],
                "x1_path": row["x1_path"],
                "x2_path": row["x2_path"],
                "vq_code": int(codes[idx]),
                "vq_mse": float(vq_mse[j]),
                "cont_mse": float(cont_mse[j]),
                "mse_advantage_cont": float(vq_mse[j] - cont_mse[j]),
            })

out_df = pd.DataFrame(rows)

# Add trajectory-level metrics.
traj_rows = []
for traj, g in out_df.groupby("traj"):
    idxs = g["idx"].to_numpy()
    traj_codes = codes[idxs]
    z_vq = Z_vq[idxs]
    z_cont = Z_cont[idxs]

    unique, counts = np.unique(traj_codes, return_counts=True)
    probs = counts / counts.sum()
    entropy = float(-(probs * np.log(probs + 1e-12)).sum())
    top_frac = float(probs.max())

    traj_rows.append({
        "traj": traj,
        "traj_num_pairs": len(g),
        "traj_label": g["label"].iloc[0],
        "traj_vq_mse_mean": float(g["vq_mse"].mean()),
        "traj_cont_mse_mean": float(g["cont_mse"].mean()),
        "traj_mse_advantage_cont": float(g["mse_advantage_cont"].mean()),
        "traj_vq_used_codes": int(len(unique)),
        "traj_vq_top_code_fraction": top_frac,
        "traj_vq_entropy": entropy,
        "traj_vq_effective_rank": effective_rank(z_vq),
        "traj_cont_effective_rank": effective_rank(z_cont),
    })

traj_df = pd.DataFrame(traj_rows)

# Failure score: high means VQ is bad relative to continuous.
traj_df["failure_score"] = (
    traj_df["traj_mse_advantage_cont"]
    + 0.01 * traj_df["traj_vq_top_code_fraction"]
    + 0.01 * (traj_df["traj_cont_effective_rank"] - traj_df["traj_vq_effective_rank"]).clip(lower=0) / 32.0
)

out_df = out_df.merge(traj_df, on="traj", how="left")

out_path = Path(args.out_csv)
out_path.parent.mkdir(parents=True, exist_ok=True)
out_df.to_csv(out_path, index=False)

traj_path = out_path.with_name(out_path.stem + "_by_traj.csv")
traj_df.sort_values("failure_score", ascending=False).to_csv(traj_path, index=False)

label_path = out_path.with_name(out_path.stem + "_by_label.csv")
label_df = traj_df.groupby("traj_label").agg(
    videos=("traj", "count"),
    mse_advantage_cont=("traj_mse_advantage_cont", "mean"),
    vq_mse=("traj_vq_mse_mean", "mean"),
    cont_mse=("traj_cont_mse_mean", "mean"),
    vq_used_codes=("traj_vq_used_codes", "mean"),
    vq_top_code_fraction=("traj_vq_top_code_fraction", "mean"),
    vq_rank=("traj_vq_effective_rank", "mean"),
    cont_rank=("traj_cont_effective_rank", "mean"),
    failure_score=("failure_score", "mean"),
).reset_index().sort_values("failure_score", ascending=False)
label_df.to_csv(label_path, index=False)

print("Saved pairwise:", out_path)
print("Saved by trajectory:", traj_path)
print("Saved by label:", label_path)
print()
print("Top VQ failure trajectories:")
print(traj_df.sort_values("failure_score", ascending=False).head(15).to_string(index=False))
print()
print("Top labels/action types:")
print(label_df.head(15).to_string(index=False))
