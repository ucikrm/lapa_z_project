import argparse
import os
import csv
import time
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from torchvision.utils import save_image

# Add project root to sys.path to allow absolute imports
import sys
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from lapa_z_compare.src.data.sthv2_pair_dataset import SthV2PairDataset
from lapa_z_compare.src.models.lapa_nsvq import LapaNSVQModel
from lapa_z_compare.src.models.adaworld_bvae import AdaWorldBVAEModel
from lapa_z_compare.src.models.univla_dino_vq import UniVLADinoVQModel
from lapa_z_compare.src.models.villax_proxy import VillaXProxyModel
from lapa_z_compare.src.models.vicreg_continuous import VicregContinuousModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True, choices=["lapa_nsvq", "adaworld_bvae", "univla_dino_vq", "villax_proxy", "vicreg"])
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--gap", type=int, default=12)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64) # Batch size per GPU
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--eval_every", type=int, default=5000)
    parser.add_argument("--max_minutes", type=float, default=-1.0, help="Stop training after this many minutes")
    
    # Method specific hyperparameters
    parser.add_argument("--num_codes", type=int, default=64) # For VQ/NSVQ models
    parser.add_argument("--beta", type=float, default=2e-4)      # For VAE beta-regularization
    parser.add_argument("--vq_beta", type=float, default=0.25)   # For VQ commitment weight
    parser.add_argument("--lambda_s", type=float, default=1.0)   # For Villa-X structural loss weight
    parser.add_argument("--lambda_var", type=float, default=1.0) # For VICReg variance loss weight
    parser.add_argument("--lambda_cov", type=float, default=0.05)# For VICReg covariance loss weight
    args = parser.parse_args()

    # DDP Setup
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        print(f"DDP process initialized. Rank: {rank}, Local Rank: {local_rank}, World Size: {world_size}")
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Single device mode. Device: {device}")

    # Set random seed based on rank for variation in data sampling
    torch.manual_seed(42 + rank)
    
    # Directories setup (only on rank 0)
    out_dir = Path(args.out_dir)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "recons").mkdir(exist_ok=True)
        (out_dir / "checkpoints").mkdir(exist_ok=True)

    # Instantiate the method-specific model
    print(f"Instantiating model for method: {args.method}...")
    if args.method == "lapa_nsvq":
        model = LapaNSVQModel(latent_dim=args.latent_dim, num_codes=args.num_codes, beta=args.vq_beta)
    elif args.method == "adaworld_bvae":
        model = AdaWorldBVAEModel(latent_dim=args.latent_dim, beta=args.beta)
    elif args.method == "univla_dino_vq":
        # Note: UniVLA requires 224x224 images for DINOv2
        model = UniVLADinoVQModel(latent_dim=args.latent_dim, num_codes=args.num_codes, beta=args.vq_beta)
    elif args.method == "villax_proxy":
        model = VillaXProxyModel(latent_dim=args.latent_dim, lambda_s=args.lambda_s, beta=args.vq_beta, num_codes=args.num_codes)
    elif args.method == "vicreg":
        model = VicregContinuousModel(latent_dim=args.latent_dim, lambda_var=args.lambda_var, lambda_cov=args.lambda_cov)
    else:
        raise ValueError(f"Unknown method {args.method}")

    model = model.to(device)
    
    # Wrap model with DDP
    if ddp:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # Precision setup
    scaler = GradScaler() if args.precision == "fp16" else None
    amp_type = torch.bfloat16 if args.precision == "bf16" else (torch.float16 if args.precision == "fp16" else torch.float32)

    # Datasets and loaders
    train_dataset = SthV2PairDataset(args.train_csv, image_size=args.image_size, gap=args.gap, split="train", compute_flow=False)
    val_dataset = SthV2PairDataset(args.val_csv, image_size=args.image_size, gap=args.gap, split="val", compute_flow=False)

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if ddp else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if ddp else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    # Set up logging CSV (only on rank 0)
    log_path = out_dir / "train_log.csv"
    log_fields = ["step", "loss"]
    
    if rank == 0:
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(log_fields)

    step = 0
    pbar = tqdm(total=args.steps, disable=(rank != 0))
    start_time = time.time()
    should_stop = False

    while step < args.steps and not should_stop:
        if ddp and train_sampler is not None:
            train_sampler.set_epoch(step)

        model.train()
        for batch in train_loader:
            if args.max_minutes > 0:
                stop_signal = torch.tensor(0.0, device=device)
                if rank == 0:
                    elapsed_min = (time.time() - start_time) / 60.0
                    if elapsed_min >= args.max_minutes:
                        stop_signal += 1.0
                if ddp:
                    dist.broadcast(stop_signal, src=0)
                if stop_signal.item() > 0.5:
                    if rank == 0:
                        print(f"Reached max_minutes limit ({args.max_minutes} min). Stopping training.")
                    should_stop = True
                    break
            x1 = batch["x1"].to(device, non_blocking=True)
            x2 = batch["x2"].to(device, non_blocking=True)
            labels = batch["label"]

            opt.zero_grad(set_to_none=True)

            with autocast(enabled=(args.precision != "fp32"), dtype=amp_type):
                # Pass labels to UniVLA model, others ignore it
                if args.method == "univla_dino_vq":
                    out = model(x1, x2, label=labels)
                else:
                    out = model(x1, x2)
                
                loss_dict = out["loss_dict"]
                loss = loss_dict["loss"]

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

            # Logging
            if step % 50 == 0 and rank == 0:
                with open(log_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    row = [step, float(loss.item())]
                    # Also log all keys in loss_dict
                    for k, v in loss_dict.items():
                        if k != "loss":
                            row.append(f"{k}:{float(v.item()):.6f}")
                    writer.writerow(row)

            # Evaluation & Visualization
            if step % args.eval_every == 0 and step > 0:
                model.eval()
                val_loss_sum = 0.0
                val_count = 0
                
                with torch.no_grad():
                    for val_batch in val_loader:
                        vx1 = val_batch["x1"].to(device, non_blocking=True)
                        vx2 = val_batch["x2"].to(device, non_blocking=True)
                        vlabels = val_batch["label"]

                        with autocast(enabled=(args.precision != "fp32"), dtype=amp_type):
                            if args.method == "univla_dino_vq":
                                vout = model(vx1, vx2, label=vlabels)
                            else:
                                vout = model(vx1, vx2)
                            vloss = vout["loss_dict"]["loss"]
                        
                        val_loss_sum += vloss.item()
                        val_count += 1
                        
                avg_val_loss = val_loss_sum / max(val_count, 1)
                
                if rank == 0:
                    print(f"\nStep {step} | Train Loss: {loss.item():.6f} | Val Loss: {avg_val_loss:.6f}")
                    
                    # Save a grid of reconstructions
                    # For UniVLA model, out["x_hat"] is feature reconstruction, not pixels, so we cannot save it directly as PNG.
                    if args.method != "univla_dino_vq":
                        recon = out["x_hat"]
                        grid = torch.cat([x1[:4], x2[:4], recon[:4]], dim=0)
                        save_image(grid, out_dir / "recons" / f"step_{step:06d}.png", nrow=4)

            # Checkpoint saving
            if step % args.save_every == 0 and step > 0 and rank == 0:
                ckpt_path = out_dir / "checkpoints" / f"step_{step:06d}.pt"
                torch.save({
                    "model": model.module.state_dict() if ddp else model.state_dict(),
                    "args": vars(args),
                    "step": step
                }, ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")

            step += 1
            if rank == 0:
                pbar.update(1)
                pbar.set_description(f"Loss: {loss.item():.4f}")

            if step >= args.steps:
                break

    pbar.close()

    # Final checkpoint
    if rank == 0:
        final_path = out_dir / "checkpoints" / "checkpoint_final.pt"
        torch.save({
            "model": model.module.state_dict() if ddp else model.state_dict(),
            "args": vars(args),
            "step": step
        }, final_path)
        print(f"Saved final checkpoint to {final_path}")

    # Cleanup DDP
    if ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
