import argparse
import sys
import time
from pathlib import Path
import torch
import torch.nn.functional as F

# Add project root to sys.path
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
    parser.add_argument("--train_csv", type=str, default="data_cache/sthv2_splits/debug/train.csv")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    # UniVLA requires 224x224 images for DINOv2
    image_size = 224 if args.method == "univla_dino_vq" else args.image_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Running Overfit Test for {args.method} on {device} ---")

    # Reconstruct model
    if args.method == "lapa_nsvq":
        model = LapaNSVQModel(latent_dim=32, num_codes=64, beta=0.25)
    elif args.method == "adaworld_bvae":
        model = AdaWorldBVAEModel(latent_dim=32, beta=2e-4)
    elif args.method == "univla_dino_vq":
        model = UniVLADinoVQModel(latent_dim=32, num_codes=64, beta=0.25)
    elif args.method == "villax_proxy":
        model = VillaXProxyModel(latent_dim=32, lambda_s=1.0, beta=0.25)
    elif args.method == "vicreg":
        model = VicregContinuousModel(latent_dim=32, lambda_var=1.0, lambda_cov=0.05)

    model = model.to(device)
    model.train()

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Load first 8 samples
    dataset = SthV2PairDataset(args.train_csv, image_size=image_size, gap=12, split="train", compute_flow=False)
    subset = torch.utils.data.Subset(dataset, list(range(8)))
    loader = torch.utils.data.DataLoader(subset, batch_size=8, shuffle=False)
    
    batch = next(iter(loader))
    x1 = batch["x1"].to(device)
    x2 = batch["x2"].to(device)
    labels = batch["label"]

    initial_rec_loss = None
    final_rec_loss = None

    for step in range(args.steps):
        opt.zero_grad()
        
        if args.method == "univla_dino_vq":
            out = model(x1, x2, label=labels)
        else:
            out = model(x1, x2)
            
        loss_dict = out["loss_dict"]
        loss = loss_dict["loss"]
        rec_loss = loss_dict["rec_loss"].item()
        
        if step == 0:
            initial_rec_loss = rec_loss
            print(f"Step {step:03d} | Total Loss: {loss.item():.6f} | Rec Loss: {rec_loss:.6f}")
            
        loss.backward()
        opt.step()
        
        if step == args.steps - 1:
            final_rec_loss = rec_loss
            print(f"Step {step:03d} | Total Loss: {loss.item():.6f} | Rec Loss: {rec_loss:.6f}")

    pct_drop = (initial_rec_loss - final_rec_loss) / initial_rec_loss * 100
    print(f"Initial Rec Loss: {initial_rec_loss:.6f} | Final Rec Loss: {final_rec_loss:.6f}")
    print(f"Reconstruction loss drop: {pct_drop:.2f}%")

    target_drop = 10.0 if args.method == "univla_dino_vq" else 20.0
    assert pct_drop >= target_drop, f"Overfitting check failed! Rec loss drop ({pct_drop:.2f}%) is less than target ({target_drop:.2f}%)"
    print(f"✓ Overfitting check PASSED for {args.method} (Drop {pct_drop:.2f}% >= {target_drop:.2f}%)\n")

if __name__ == "__main__":
    main()
