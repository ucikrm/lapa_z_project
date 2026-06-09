import argparse
from pathlib import Path
import csv

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from frame_pair_dataset import FramePairDataset
from phase1_models import VQPhase1Model


parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, default="../data/tiny_debug")
parser.add_argument("--out_dir", type=str, default="../outputs/vq_lapa")
parser.add_argument("--image_size", type=int, default=128)
parser.add_argument("--gap", type=int, default=1)
parser.add_argument("--latent_dim", type=int, default=32)
parser.add_argument("--num_codes", type=int, default=64)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument("--steps", type=int, default=5000)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--save_every", type=int, default=500)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "recons").mkdir(exist_ok=True)

dataset = FramePairDataset(args.data_root, image_size=args.image_size, gap=args.gap)
loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, pin_memory=True, drop_last=True)

model = VQPhase1Model(latent_dim=args.latent_dim, num_codes=args.num_codes).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

log_path = out_dir / "train_log.csv"
with open(log_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["step", "loss_total", "loss_rec", "loss_vq", "used_codes"])

step = 0
pbar = tqdm(total=args.steps)

while step < args.steps:
    for batch in loader:
        x1 = batch["x1"].to(device, non_blocking=True)
        x2 = batch["x2"].to(device, non_blocking=True)

        out = model(x1, x2)

        rec_loss = F.mse_loss(out["x2_hat"], x2)
        vq_loss = out["vq_loss"]
        loss = rec_loss + vq_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        used_codes = out["indices"].unique().numel()

        if step % 50 == 0:
            with open(log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([step, float(loss.item()), float(rec_loss.item()),
                                 float(vq_loss.item()), int(used_codes)])

        if step % args.save_every == 0:
            torch.save({"model": model.state_dict(), "args": vars(args), "step": step},
                       out_dir / "checkpoint.pt")
            grid = torch.cat([x1[:4], x2[:4], out["x2_hat"][:4]], dim=0)
            save_image(grid, out_dir / "recons" / f"step_{step:06d}.png", nrow=4)

        step += 1
        pbar.update(1)
        if step >= args.steps:
            break

pbar.close()

torch.save({"model": model.state_dict(), "args": vars(args), "step": step},
           out_dir / "checkpoint_final.pt")

print("Done VQ training:", out_dir)
