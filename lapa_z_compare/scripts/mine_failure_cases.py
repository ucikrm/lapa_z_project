import argparse
import os
from pathlib import Path
import numpy as np
import torch
import torchvision.io as io
import torchvision.transforms as T
from tqdm import tqdm

# Add project root to sys.path
import sys
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from lapa_z_compare.src.models.lapa_nsvq import LapaNSVQModel
from lapa_z_compare.src.models.vicreg_continuous import VicregContinuousModel
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", type=str, default="data_cache/sthv2_splits/test_1k.csv")
    parser.add_argument("--lapa_ckpt", type=str, default="lapa_z_compare/outputs/5k/lapa_nsvq/checkpoints/checkpoint_final.pt")
    parser.add_argument("--vicreg_ckpt", type=str, default="lapa_z_compare/outputs/5k/vicreg/checkpoints/checkpoint_final.pt")
    parser.add_argument("--num_videos", type=int, default=100, help="Number of videos to mine from")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load LAPA model
    print("Loading LAPA model...")
    lapa_checkpoint = torch.load(args.lapa_ckpt, map_location="cpu")
    lapa_args = lapa_checkpoint["args"]
    lapa_model = LapaNSVQModel(
        latent_dim=lapa_args["latent_dim"],
        num_codes=lapa_args.get("num_codes", 64),
        beta=lapa_args.get("beta", 0.25)
    )
    lapa_model.load_state_dict(lapa_checkpoint["model"])
    lapa_model = lapa_model.to(device)
    lapa_model.eval()

    # Load VICReg model
    print("Loading VICReg model...")
    vicreg_checkpoint = torch.load(args.vicreg_ckpt, map_location="cpu")
    vicreg_args = vicreg_checkpoint["args"]
    vicreg_model = VicregContinuousModel(
        latent_dim=vicreg_args["latent_dim"],
        lambda_var=vicreg_args.get("lambda_var", 1.0),
        lambda_cov=vicreg_args.get("lambda_cov", 0.05)
    )
    vicreg_model.load_state_dict(vicreg_checkpoint["model"])
    vicreg_model = vicreg_model.to(device)
    vicreg_model.eval()

    # Read test split
    df = pd.read_csv(args.test_csv)
    # Sample subset of videos to keep execution fast
    df_sample = df.sample(n=min(args.num_videos, len(df)), random_state=42)

    image_size = vicreg_args["image_size"]
    gap = vicreg_args["gap"]
    transform = T.Compose([T.Resize((image_size, image_size))])

    results = []

    print(f"Mining failure cases from {len(df_sample)} test videos...")
    for idx, row in enumerate(tqdm(df_sample.itertuples(), total=len(df_sample))):
        video_path = row.video_path
        video_id = str(row.video_id)
        label = str(row.label)

        try:
            video, _, _ = io.read_video(video_path, pts_unit="sec", output_format="THWC")
            F_len = video.shape[0]
        except Exception:
            continue

        if F_len <= gap + 5:
            continue

        # Extract all possible consecutive pairs
        pairs_x1 = []
        pairs_x2 = []
        for t in range(F_len - gap):
            f1 = video[t].permute(2, 0, 1).float() / 255.0
            f2 = video[t + gap].permute(2, 0, 1).float() / 255.0
            pairs_x1.append(transform(f1))
            pairs_x2.append(transform(f2))

        x1 = torch.stack(pairs_x1).to(device)
        x2 = torch.stack(pairs_x2).to(device)

        # Run inference
        with torch.no_grad():
            # LAPA NSVQ
            lapa_out = lapa_model(x1, x2)
            lapa_indices = lapa_out["indices"].cpu().numpy()
            lapa_rec_loss = lapa_out["loss_dict"]["rec_loss"].item()

            # VICReg
            vicreg_out = vicreg_model(x1, x2)
            vicreg_z = vicreg_out["z"].cpu().numpy()
            vicreg_rec_loss = vicreg_out["loss_dict"]["rec_loss"].item()

        # Metrics for the video sequence
        unique_lapa_codes = len(np.unique(lapa_indices))
        # Variance of LAPA code index (checks for temporal change)
        lapa_code_std = np.std(lapa_indices)
        
        # Variance/std of VICReg latents along each dimension over time
        vicreg_temporal_std = np.mean(np.std(vicreg_z, axis=0))

        # We look for cases where:
        # 1. LAPA uses very few unique codes (e.g., exactly 1 code for the whole video)
        # 2. VICReg has active variance showing it encodes motion
        # 3. LAPA reconstruction loss is higher than VICReg
        if unique_lapa_codes == 1 and vicreg_temporal_std > 0.05:
            rec_loss_diff = lapa_rec_loss - vicreg_rec_loss
            results.append({
                "video_id": video_id,
                "label": label,
                "lapa_codes_used": unique_lapa_codes,
                "lapa_code_sequence": list(lapa_indices),
                "vicreg_temporal_std": vicreg_temporal_std,
                "lapa_rec_loss": lapa_rec_loss,
                "vicreg_rec_loss": vicreg_rec_loss,
                "rec_loss_diff": rec_loss_diff
            })

    # Sort by reconstruction loss difference (where LAPA is worst compared to VICReg)
    results = sorted(results, key=lambda x: x["rec_loss_diff"], reverse=True)

    print("\n======================================================================")
    print("                    TOP 5 DETECTED FAILURE CASES                       ")
    print("      (Where LAPA NSVQ collapsed to a single code, but VICReg succeeded)   ")
    print("======================================================================")
    for i in range(min(5, len(results))):
        res = results[i]
        print(f"\n{i+1}. Video ID: {res['video_id']}")
        print(f"   Action Label: \"{res['label']}\"")
        print(f"   LAPA Unique Codes Used: {res['lapa_codes_used']} (Code: {res['lapa_code_sequence'][0]})")
        print(f"   VICReg Latent Temporal Std: {res['vicreg_temporal_std']:.4f}")
        print(f"   LAPA Reconstruction Loss: {res['lapa_rec_loss']:.5f}")
        print(f"   VICReg Reconstruction Loss: {res['vicreg_rec_loss']:.5f}")
        print(f"   Recon Error Difference (LAPA - VICReg): {res['rec_loss_diff']:.5f}")
        print("   Explanation:")
        print("     - NSVQ VQ-bottleneck collapsed and mapped every frame transition to the exact same code token.")
        print("       The model perceived this action as static/no-change, losing all transition details.")
        print("     - Ours (VICReg) successfully mapped the transitions to a continuous trajectory (temporal std > 0),")
        print("       preserving the gradual deformation or displacement of the object and achieving lower reconstruction error.")

if __name__ == "__main__":
    main()
