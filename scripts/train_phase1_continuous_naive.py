import argparse
from pathlib import Path
import csv

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from frame_pair_dataset import FramePairDataset
from phase1_models import NaiveContinuousPhase1Model


parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, required=True)
parser.add_argument("--out_dir", type=str, required=True)
parser.add_argument("--image_size", type=int, default=128)
parser.add_argument("--gap", type=int, default=1)
parser.add_argument("--latent_dim", type=int, default=32)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument("--steps", type=int, default=10000)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--save_every", type=int, default=500)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "recons").mkdir(exist_ok=True)

dataset = FramePairDataset(args.data_root, image_size=args.image_size, gap=args.gap)
loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True,
    drop_last=True,
)

model = NaiveContinuousPhase1Model(latent_dim=args.latent_dim).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

log_path = out_dir / "train_log.csv"
with open(log_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "step",
        "loss_total",
        "loss_rec",
        "z_abs_mean",
        "z_std_mean",
        "z_std_min",
        "z_std_max",
    ])

step = 0
pbar = tqdm(total=args.steps)

while step < args.steps:
    for batch in loader:
        x1 = batch["x1"].to(device, non_blocking=True)
        x2 = batch["x2"].to(device, non_blocking=True)

        out = model(x1, x2)
        z = out["z"]

        rec_loss = F.mse_loss(out["x2_hat"], x2)
        loss = rec_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        z_det = z.detach().float()
        z_std = z_det.std(dim=0)

        if step % 50 == 0:
            with open(log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    step,
                    float(loss.item()),
                    float(rec_loss.item()),
                    float(z_det.abs().mean().item()),
                    float(z_std.mean().item()),
                    float(z_std.min().item()),
                    float(z_std.max().item()),
                ])

        if step % args.save_every == 0:
            torch.save(
                {"model": model.state_dict(), "args": vars(args), "step": step},
                out_dir / "checkpoint.pt",
            )

            grid = torch.cat([x1[:4], x2[:4], out["x2_hat"][:4]], dim=0)
            save_image(grid, out_dir / "recons" / f"step_{step:06d}.png", nrow=4)

        step += 1
        pbar.update(1)

        if step >= args.steps:
            break

pbar.close()

torch.save(
    {"model": model.state_dict(), "args": vars(args), "step": step},
    out_dir / "checkpoint_final.pt",
)

print("Done naive continuous training:", out_dir)
