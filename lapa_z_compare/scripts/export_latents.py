import argparse
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to sys.path
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
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--test_csv", type=str, required=True, help="Path to test split CSV")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save the latents npz")
    parser.add_argument("--out_name", type=str, default="latents_test.npz", help="Output npz filename")
    parser.add_argument("--split", type=str, default="test", help="Dataset split tag for sampling")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    checkpoint = torch.load(args.ckpt, map_location="cpu")
    train_args = checkpoint["args"]
    method = train_args["method"]
    latent_dim = train_args["latent_dim"]
    image_size = train_args["image_size"]
    gap = train_args["gap"]

    print(f"Loaded checkpoint from {args.ckpt}")
    print(f"Method: {method} | Latent dim: {latent_dim} | Gap: {gap}")

    # Reconstruct model
    if method == "lapa_nsvq":
        model = LapaNSVQModel(latent_dim=latent_dim, num_codes=train_args.get("num_codes", 64), beta=train_args.get("vq_beta", train_args.get("beta", 0.25)))
    elif method == "adaworld_bvae":
        model = AdaWorldBVAEModel(latent_dim=latent_dim, beta=train_args.get("beta", 2e-4))
    elif method == "univla_dino_vq":
        model = UniVLADinoVQModel(latent_dim=latent_dim, num_codes=train_args.get("num_codes", 64), beta=train_args.get("vq_beta", train_args.get("beta", 0.25)))
    elif method == "villax_proxy":
        model = VillaXProxyModel(latent_dim=latent_dim, lambda_s=train_args.get("lambda_s", 1.0), beta=train_args.get("vq_beta", 0.25), num_codes=train_args.get("num_codes", 64))
    elif method == "vicreg":
        model = VicregContinuousModel(latent_dim=latent_dim, lambda_var=train_args.get("lambda_var", 1.0), lambda_cov=train_args.get("lambda_cov", 0.05))
    else:
        raise ValueError(f"Unknown method {method}")

    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()

    # Load test dataset
    dataset = SthV2PairDataset(args.test_csv, image_size=image_size, gap=gap, split=args.split, compute_flow=False)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    def compute_per_sample_mse(pred, target):
        dims = tuple(range(1, pred.ndim))
        return torch.mean((pred - target) ** 2, dim=dims)

    latents = []
    video_ids = []
    labels = []
    templates = []
    ts = []
    num_frames_list = []
    recon_losses = []
    recon_losses_zero = []
    recon_losses_random = []
    indices_list = []

    print("Running inference and exporting latents...")
    with torch.no_grad():
        for batch in tqdm(loader):
            x1 = batch["x1"].to(device)
            x2 = batch["x2"].to(device)
            batch_labels = batch["label"]
            batch_templates = batch["template"]
            batch_vids = batch["video_id"]
            batch_t = batch["t"]
            batch_num_frames = batch["num_frames"]

            # Run forward pass
            if method == "univla_dino_vq":
                out = model(x1, x2, label=batch_labels)
                target = model.get_dino_features(x2)
                z = out["z"]
                
                # Zero z reconstruction
                z_zero = torch.zeros_like(z)
                x_hat_zero = model.decode(x1, z_zero, label=batch_labels)
                
                # Random/swapped z reconstruction
                z_random = torch.roll(z, shifts=1, dims=0)
                x_hat_random = model.decode(x1, z_random, label=batch_labels)
            else:
                out = model(x1, x2)
                target = x2
                z = out["z"]
                
                # Zero z reconstruction
                z_zero = torch.zeros_like(z)
                x_hat_zero = model.decode(x1, z_zero)
                
                # Random/swapped z reconstruction
                z_random = torch.roll(z, shifts=1, dims=0)
                x_hat_random = model.decode(x1, z_random)

            x_hat = out["x_hat"]
            
            # Compute per-sample MSEs
            rec_loss_sample = compute_per_sample_mse(x_hat, target)
            rec_loss_zero_sample = compute_per_sample_mse(x_hat_zero, target)
            rec_loss_random_sample = compute_per_sample_mse(x_hat_random, target)

            latents.append(z.cpu().numpy())
            recon_losses.extend(rec_loss_sample.cpu().numpy())
            recon_losses_zero.extend(rec_loss_zero_sample.cpu().numpy())
            recon_losses_random.extend(rec_loss_random_sample.cpu().numpy())
            
            if "indices" in out:
                indices_list.extend(out["indices"].cpu().numpy())
            else:
                indices_list.extend([-1] * z.shape[0])
            
            video_ids.extend(batch_vids)
            labels.extend(batch_labels)
            templates.extend(batch_templates)
            ts.extend(batch_t.cpu().numpy())
            num_frames_list.extend(batch_num_frames.cpu().numpy())

    # Concatenate results
    latents = np.concatenate(latents, axis=0)
    recon_losses = np.array(recon_losses)
    recon_losses_zero = np.array(recon_losses_zero)
    recon_losses_random = np.array(recon_losses_random)
    indices_arr = np.array(indices_list)
    video_ids = np.array(video_ids)
    labels = np.array(labels)
    templates = np.array(templates)
    ts = np.array(ts)
    num_frames_list = np.array(num_frames_list)

    # Save to disk
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out_name
    
    np.savez_compressed(
        out_path,
        z=latents,
        video_id=video_ids,
        label=labels,
        template=templates,
        t=ts,
        num_frames=num_frames_list,
        recon_loss=recon_losses,
        recon_loss_zero=recon_losses_zero,
        recon_loss_random=recon_losses_random,
        indices=indices_arr
    )

    print(f"Export completed! Latents shape: {latents.shape}")
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
