import argparse
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

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
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--npz", type=str, required=True)
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--train_csv", type=str, required=True)
    args = parser.parse_args()

    print(f"\n========== Verifying Smoke Test for {args.method} ==========")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load checkpoint and check parameters
    checkpoint = torch.load(args.ckpt, map_location="cpu")
    train_args = checkpoint["args"]
    latent_dim = train_args["latent_dim"]
    
    # 2. Reconstruct model
    if args.method == "lapa_nsvq":
        model = LapaNSVQModel(latent_dim=latent_dim, num_codes=train_args.get("num_codes", 64), beta=train_args.get("vq_beta", 0.25))
    elif args.method == "adaworld_bvae":
        model = AdaWorldBVAEModel(latent_dim=latent_dim, beta=train_args.get("beta", 2e-4))
    elif args.method == "univla_dino_vq":
        model = UniVLADinoVQModel(latent_dim=latent_dim, num_codes=train_args.get("num_codes", 64), beta=train_args.get("vq_beta", 0.25))
    elif args.method == "villax_proxy":
        model = VillaXProxyModel(latent_dim=latent_dim, lambda_s=train_args.get("lambda_s", 1.0), beta=train_args.get("vq_beta", 0.25))
    elif args.method == "vicreg":
        model = VicregContinuousModel(latent_dim=latent_dim, lambda_var=train_args.get("lambda_var", 1.0), lambda_cov=train_args.get("lambda_cov", 0.05))
    else:
        raise ValueError(f"Unknown method {args.method}")

    model.load_state_dict(checkpoint["model"])
    model = model.to(device)

    # Check UniVLA decoder channels (no appearance shortcut)
    if args.method == "univla_dino_vq":
        in_ch = model.feature_decoder[0].in_channels
        assert in_ch == 256, f"UniVLADinoVQModel decoder should have 256 input channels (got {in_ch})."
        print("✓ verified: UniVLA decoder has no appearance shortcut (in_channels=256).")

    # 3. Test forward/backward gradients
    model.train()
    dataset = SthV2PairDataset(args.train_csv, image_size=96, gap=12, split="train", compute_flow=False)
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=2)))
    x1 = batch["x1"].to(device)
    x2 = batch["x2"].to(device)
    labels = batch["label"]

    if args.method == "univla_dino_vq":
        out = model(x1, x2, label=labels)
    else:
        out = model(x1, x2)

    loss = out["loss_dict"]["loss"]
    assert torch.isfinite(loss), "Loss is not finite!"
    
    loss.backward()

    # Verify that requires_grad parameters have non-zero gradients
    grad_ok = True
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                print(f"✗ ERROR: Parameter {name} has no gradient!")
                grad_ok = False
            elif param.grad.abs().sum() == 0:
                print(f"✗ ERROR: Parameter {name} gradient is zero!")
                grad_ok = False
    
    assert grad_ok, "Gradient checks failed!"
    print("✓ verified: All trainable parameters received non-zero gradients.")

    # 4. Check exported latents npz
    data = np.load(args.npz)
    assert "z" in data, "npz missing 'z'"
    assert "recon_loss" in data, "npz missing 'recon_loss'"
    assert "recon_loss_zero" in data, "npz missing 'recon_loss_zero'"
    assert "recon_loss_random" in data, "npz missing 'recon_loss_random'"
    
    z = data["z"]
    assert z.shape[1] == latent_dim, f"Latent shape mismatch! Expected dim {latent_dim}, got {z.shape[1]}"
    print(f"✓ verified: Exported z shape: {z.shape}")

    # 5. Check metrics CSV
    df = pd.read_csv(args.csv)
    row = df.iloc[0]
    
    print("\nMetrics summary:")
    for k, v in row.items():
        print(f"  {k}: {v}")

    # Check discrete vs continuous consistency
    is_discrete = args.method in ["lapa_nsvq", "univla_dino_vq", "villax_proxy"]
    if is_discrete:
        assert row["code_perplexity"] > 0, f"Discrete method {args.method} has invalid perplexity {row['code_perplexity']}"
        assert 0 <= row["dead_code_pct"] <= 100, f"Discrete method {args.method} has invalid dead_code_pct {row['dead_code_pct']}"
        print(f"✓ verified: Discrete metrics are valid (perplexity={row['code_perplexity']:.2f}, dead={row['dead_code_pct']:.2f}%)")
    else:
        assert row["code_perplexity"] == -1.0, f"Continuous method {args.method} should have perplexity -1.0 (got {row['code_perplexity']})"
        assert row["dead_code_pct"] == -1.0, f"Continuous method {args.method} should have dead_code_pct -1.0 (got {row['dead_code_pct']})"
        print("✓ verified: Continuous metrics are set to -1.0 (will map to N/A).")

    print(f"========== Verification SUCCESS for {args.method} ==========\n")

if __name__ == "__main__":
    main()
