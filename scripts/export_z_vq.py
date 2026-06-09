import argparse
from pathlib import Path
import csv

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from frame_pair_dataset import FramePairDataset
from phase1_models import VQPhase1Model


parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, default="../data/tiny_debug")
parser.add_argument("--checkpoint", type=str, default="../outputs/vq_lapa/checkpoint_final.pt")
parser.add_argument("--out_dir", type=str, default="../z_exports/vq_lapa")
parser.add_argument("--image_size", type=int, default=128)
parser.add_argument("--gap", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--num_workers", type=int, default=2)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

ckpt = torch.load(args.checkpoint, map_location="cpu")
train_args = ckpt["args"]

model = VQPhase1Model(
    latent_dim=train_args["latent_dim"],
    num_codes=train_args["num_codes"],
).to(device)

model.load_state_dict(ckpt["model"])
model.eval()

dataset = FramePairDataset(args.data_root, image_size=args.image_size, gap=args.gap)
loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)

all_z = []
all_z_e = []
all_indices = []
rows = []
idx_global = 0

with torch.no_grad():
    for batch in tqdm(loader):
        x1 = batch["x1"].to(device, non_blocking=True)
        x2 = batch["x2"].to(device, non_blocking=True)

        out = model(x1, x2)

        z = out["z"].detach().cpu().numpy()
        z_e = out["z_e"].detach().cpu().numpy()
        indices = out["indices"].detach().cpu().numpy()

        all_z.append(z)
        all_z_e.append(z_e)
        all_indices.append(indices)

        for i in range(len(indices)):
            rows.append([idx_global, batch["x1_path"][i], batch["x2_path"][i], int(indices[i])])
            idx_global += 1

Z = np.concatenate(all_z, axis=0)
Z_e = np.concatenate(all_z_e, axis=0)
indices = np.concatenate(all_indices, axis=0)

np.save(out_dir / "z_quantized.npy", Z)
np.save(out_dir / "z_encoder.npy", Z_e)
np.save(out_dir / "indices.npy", indices)

with open(out_dir / "metadata.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["idx", "x1_path", "x2_path", "code_index"])
    writer.writerows(rows)

print("Saved:")
print(out_dir / "z_quantized.npy", Z.shape)
print(out_dir / "z_encoder.npy", Z_e.shape)
print(out_dir / "indices.npy", indices.shape)
