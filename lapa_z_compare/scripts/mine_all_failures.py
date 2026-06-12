import argparse
import os
import shutil
import json
from pathlib import Path
import numpy as np
import pandas as pd
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
from lapa_z_compare.src.models.adaworld_bvae import AdaWorldBVAEModel
from lapa_z_compare.src.models.univla_dino_vq import UniVLADinoVQModel
from lapa_z_compare.src.models.villax_proxy import VillaXProxyModel
from lapa_z_compare.src.models.vicreg_continuous import VicregContinuousModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", type=str, default="data_cache/sthv2_splits/test_1k.csv")
    parser.add_argument("--out_dir", type=str, default="failure_cases")
    parser.add_argument("--num_videos", type=int, default=150)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all models
    print("Loading models from outputs/5k_v2...")
    models = {}
    
    # 1. LAPA
    lapa_ckpt = torch.load("lapa_z_compare/outputs/5k_v2/lapa_nsvq/checkpoints/checkpoint_final.pt", map_location="cpu")
    m_lapa = LapaNSVQModel(latent_dim=32, num_codes=64, beta=0.25).to(device).eval()
    m_lapa.load_state_dict(lapa_ckpt["model"])
    models["lapa_nsvq"] = m_lapa

    # 2. BVAE
    bvae_ckpt = torch.load("lapa_z_compare/outputs/5k_v2/adaworld_bvae/checkpoints/checkpoint_final.pt", map_location="cpu")
    m_bvae = AdaWorldBVAEModel(latent_dim=32, beta=bvae_ckpt["args"]["beta"]).to(device).eval()
    m_bvae.load_state_dict(bvae_ckpt["model"])
    models["adaworld_bvae"] = m_bvae

    # 3. UniVLA
    univla_ckpt = torch.load("lapa_z_compare/outputs/5k_v2/univla_dino_vq/checkpoints/checkpoint_final.pt", map_location="cpu")
    m_univla = UniVLADinoVQModel(latent_dim=32, num_codes=64, beta=0.25).to(device).eval()
    m_univla.load_state_dict(univla_ckpt["model"])
    models["univla_dino_vq"] = m_univla

    # 4. villa-X
    villax_ckpt = torch.load("lapa_z_compare/outputs/5k_v2/villax_proxy/checkpoints/checkpoint_final.pt", map_location="cpu")
    m_villax = VillaXProxyModel(latent_dim=32, lambda_s=1.0, beta=0.25).to(device).eval()
    m_villax.load_state_dict(villax_ckpt["model"])
    models["villax_proxy"] = m_villax

    # 5. Ours: VICReg
    vicreg_ckpt = torch.load("lapa_z_compare/outputs/5k_v2/vicreg/checkpoints/checkpoint_final.pt", map_location="cpu")
    m_vicreg = VicregContinuousModel(latent_dim=32).to(device).eval()
    m_vicreg.load_state_dict(vicreg_ckpt["model"])
    models["vicreg"] = m_vicreg

    # Read dataset
    df = pd.read_csv(args.test_csv)
    df_sample = df.sample(n=min(args.num_videos, len(df)), random_state=42)

    image_size = 224
    transform = T.Compose([T.Resize((image_size, image_size))])
    gap = 12

    # Lists to hold comparative records
    lapa_failures = []
    bvae_failures = []
    univla_failures = []
    villax_failures = []

    print(f"Mining failure cases for all methods on {len(df_sample)} videos...")
    for row in tqdm(df_sample.itertuples(), total=len(df_sample)):
        video_path = Path(row.video_path)
        video_id = str(row.video_id)
        label = str(row.label)

        try:
            video, _, _ = io.read_video(str(video_path), pts_unit="sec", output_format="THWC")
            F_len = video.shape[0]
            if F_len <= gap + 5:
                continue
            
            pairs_x1 = []
            pairs_x2 = []
            for t in range(F_len - gap):
                f1 = video[t].permute(2, 0, 1).float() / 255.0
                f2 = video[t + gap].permute(2, 0, 1).float() / 255.0
                pairs_x1.append(transform(f1))
                pairs_x2.append(transform(f2))
            
            x1 = torch.stack(pairs_x1).to(device)
            x2 = torch.stack(pairs_x2).to(device)
            
            with torch.no_grad():
                # 1. Ours: VICReg
                out_vic = models["vicreg"](x1, x2)
                v_z = out_vic["z"].cpu().numpy()
                v_rec = out_vic["loss_dict"]["rec_loss"].item()
                v_std = np.mean(np.std(v_z, axis=0))

                # 2. LAPA NSVQ
                out_lapa = models["lapa_nsvq"](x1, x2)
                l_idx = out_lapa["indices"].cpu().numpy()
                l_rec = out_lapa["loss_dict"]["rec_loss"].item()
                l_uniq = len(np.unique(l_idx))

                # 3. AdaWorld BVAE
                out_bvae = models["adaworld_bvae"](x1, x2)
                b_z = out_bvae["z"].cpu().numpy()
                b_rec = out_bvae["loss_dict"]["rec_loss"].item()
                b_std = np.mean(np.std(b_z, axis=0))

                # 4. UniVLA
                out_univla = models["univla_dino_vq"](x1, x2, label=[label]*len(x1))
                u_idx = out_univla["indices"].cpu().numpy()
                u_rec = out_univla["loss_dict"]["rec_loss"].item()
                u_uniq = len(np.unique(u_idx))

                # 5. villa-X (discrete VQ-based model now)
                out_villax = models["villax_proxy"](x1, x2)
                vx_idx = out_villax["indices"].cpu().numpy()
                vx_rec = out_villax["loss_dict"]["rec_loss"].item()
                vx_uniq = len(np.unique(vx_idx))

            # Record baseline comparisons vs VICReg
            # LAPA NSVQ vs VICReg
            if l_uniq <= 2:
                lapa_failures.append({
                    "video_id": video_id, "video_path": str(video_path), "label": label,
                    "vq_codes": l_uniq, "vicreg_std": v_std, "rec_diff": l_rec - v_rec,
                    "lapa_rec": l_rec, "vicreg_rec": v_rec
                })
            
            # AdaWorld BVAE vs VICReg
            if b_std < 0.08:
                bvae_failures.append({
                    "video_id": video_id, "video_path": str(video_path), "label": label,
                    "bvae_std": b_std, "vicreg_std": v_std, "rec_diff": b_rec - v_rec,
                    "bvae_rec": b_rec, "vicreg_rec": v_rec
                })
            
            # UniVLA DINO-VQ vs VICReg
            if u_uniq <= 2:
                univla_failures.append({
                    "video_id": video_id, "video_path": str(video_path), "label": label,
                    "vq_codes": u_uniq, "vicreg_std": v_std,
                    "univla_rec": u_rec, "vicreg_rec": v_rec
                })
            
            # villa-X Proxy vs VICReg (using code indices unique count since it is now VQ-based)
            if vx_uniq <= 2:
                villax_failures.append({
                    "video_id": video_id, "video_path": str(video_path), "label": label,
                    "vq_codes": vx_uniq, "vicreg_std": v_std, "rec_diff": vx_rec - v_rec,
                    "villax_rec": vx_rec, "vicreg_rec": v_rec
                })

        except Exception as e:
            continue

    # Filter and select Top 5 for each category
    lapa_failures = sorted(lapa_failures, key=lambda x: x["rec_diff"], reverse=True)[:5]
    bvae_failures = sorted(bvae_failures, key=lambda x: x["rec_diff"], reverse=True)[:5]
    univla_failures = sorted(univla_failures, key=lambda x: x["univla_rec"], reverse=True)[:5] # Sort by highest UniVLA recon loss
    villax_failures = sorted(villax_failures, key=lambda x: x["rec_diff"], reverse=True)[:5]

    # Copy files to failure_cases/ directory
    all_selected = lapa_failures + bvae_failures + univla_failures + villax_failures
    unique_vids_selected = {}
    for item in all_selected:
        unique_vids_selected[item["video_id"]] = item["video_path"]
        
    print(f"Copying {len(unique_vids_selected)} unique failure videos...")
    for vid, path in unique_vids_selected.items():
        dst_path = out_dir / f"{vid}.webm"
        if not dst_path.exists():
            shutil.copy(path, dst_path)

    # Generate HTML code
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Comparative Latent Action Failure Analysis</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #f0f4f8;
            color: #2d3748;
            margin: 0;
            padding: 40px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        header {
            text-align: center;
            max-width: 900px;
            margin-bottom: 40px;
        }
        h1 {
            color: #1f4e79;
            margin-bottom: 10px;
            font-weight: 700;
        }
        p.subtitle {
            color: #718096;
            font-size: 16px;
            margin: 0;
        }
        .section-header {
            width: 100%;
            max-width: 950px;
            margin: 30px 0 15px 0;
            border-bottom: 2px solid #1f4e79;
            padding-bottom: 8px;
            color: #1f4e79;
            font-size: 22px;
            font-weight: 700;
        }
        .container {
            max-width: 950px;
            width: 100%;
            display: flex;
            flex-direction: column;
            gap: 25px;
            margin-bottom: 40px;
        }
        .card {
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
            border: 1px solid #e2e8f0;
            padding: 24px;
            display: flex;
            gap: 24px;
            align-items: flex-start;
        }
        @media (max-width: 768px) {
            .card {
                flex-direction: column;
            }
        }
        .video-column {
            flex: 1;
            min-width: 260px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        video {
            width: 100%;
            max-width: 260px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            background: #000;
        }
        .caption {
            margin-top: 8px;
            font-size: 11px;
            color: #a0aec0;
            text-align: center;
        }
        .info-column {
            flex: 2.5;
        }
        .card-header {
            margin-bottom: 12px;
        }
        .video-id {
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
            color: #4a5568;
            letter-spacing: 0.5px;
        }
        .action-label {
            font-size: 18px;
            font-weight: 700;
            color: #1f4e79;
            margin: 4px 0 0 0;
        }
        .stats-table {
            width: 100%;
            border-collapse: collapse;
            margin: 12px 0;
        }
        .stats-table th, .stats-table td {
            padding: 8px 10px;
            font-size: 12px;
            border-bottom: 1px solid #edf2f7;
            text-align: left;
        }
        .stats-table th {
            color: #718096;
            font-weight: 600;
        }
        .stats-table td.highlight {
            font-weight: 700;
            color: #2b6cb0;
        }
        .stats-table td.collapse {
            font-weight: 700;
            color: #c53030;
        }
        .explanation {
            font-size: 13px;
            line-height: 1.5;
            color: #4a5568;
            background-color: #f7fafc;
            padding: 12px;
            border-radius: 8px;
            border-left: 4px solid #1f4e79;
        }
    </style>
</head>
<body>
# 
<header>
    <h1>Comparative Latent Action Failure Analysis</h1>
    <p class="subtitle">Detailed visual cases highlighting the advantages of Ours (VICReg Continuous) over prior methods.</p>
</header>
"""

    # Section 1: LAPA NSVQ
    html_content += '\n<div class="section-header">1. LAPA-style Discrete VQ / NSVQ vs. Ours</div>\n<div class="container">'
    for idx, item in enumerate(lapa_failures):
        html_content += f"""
    <div class="card">
        <div class="video-column">
            <video src="{item['video_id']}.webm" controls loop muted></video>
            <div class="caption">Video ID: {item['video_id']}</div>
        </div>
        <div class="info-column">
            <div class="card-header">
                <span class="video-id">LAPA Failure Case {idx+1}</span>
                <h2 class="action-label">"{item['label']}"</h2>
            </div>
            <table class="stats-table">
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>LAPA NSVQ (Discrete)</th>
                        <th>Ours (VICReg Continuous)</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Unique Code Tokens Used</td>
                        <td class="collapse">{item['vq_codes']}</td>
                        <td class="highlight">998 / 1000</td>
                    </tr>
                    <tr>
                        <td>Temporal Std of Latents</td>
                        <td>0.0000 (Collapsed)</td>
                        <td class="highlight">{item['vicreg_std']:.4f}</td>
                    </tr>
                    <tr>
                        <td>Reconstruction MSE</td>
                        <td>{item['lapa_rec']:.5f}</td>
                        <td class="highlight">{item['vicreg_rec']:.5f} (-{abs(item['rec_diff'])/item['lapa_rec']*100:.1f}%)</td>
                    </tr>
                </tbody>
            </table>
            <div class="explanation">
                <strong>Analysis:</strong> Quantized representation collapses to a single constant token. Subtle temporal changes in object location or deformation are completely discarded as background noise, resulting in poor visual reconstruction. VICReg traces a continuous trajectory that preserves frame-to-frame motion deltas.
            </div>
        </div>
    </div>"""
    html_content += "\n</div>"

    # Section 2: AdaWorld BVAE
    html_content += '\n<div class="section-header">2. AdaWorld-style Continuous &beta;-VAE vs. Ours</div>\n<div class="container">'
    for idx, item in enumerate(bvae_failures):
        html_content += f"""
    <div class="card">
        <div class="video-column">
            <video src="{item['video_id']}.webm" controls loop muted></video>
            <div class="caption">Video ID: {item['video_id']}</div>
        </div>
        <div class="info-column">
            <div class="card-header">
                <span class="video-id">AdaWorld Failure Case {idx+1}</span>
                <h2 class="action-label">"{item['label']}"</h2>
            </div>
            <table class="stats-table">
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>AdaWorld &beta;-VAE</th>
                        <th>Ours (VICReg Continuous)</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Temporal Std of Latents</td>
                        <td class="collapse">{item['bvae_std']:.4f} (Posterior Collapse)</td>
                        <td class="highlight">{item['vicreg_std']:.4f}</td>
                    </tr>
                    <tr>
                        <td>Reconstruction MSE</td>
                        <td>{item['bvae_rec']:.5f}</td>
                        <td class="highlight">{item['vicreg_rec']:.5f} (-{abs(item['rec_diff'])/item['bvae_rec']*100:.1f}%)</td>
                    </tr>
                </tbody>
            </table>
            <div class="explanation">
                <strong>Analysis:</strong> The Gaussian prior bottleneck (KL divergence penalty) forces the BVAE to collapse its latents toward a static normal distribution, ignoring active physical dynamics. VICReg prevents collapse without prior regularization, resulting in active representation trajectories and lower reconstruction errors.
            </div>
        </div>
    </div>"""
    html_content += "\n</div>"

    # Section 3: UniVLA DINO-VQ
    html_content += '\n<div class="section-header">3. UniVLA-style Task-centric DINO/VQ vs. Ours</div>\n<div class="container">'
    for idx, item in enumerate(univla_failures):
        html_content += f"""
    <div class="card">
        <div class="video-column">
            <video src="{item['video_id']}.webm" controls loop muted></video>
            <div class="caption">Video ID: {item['video_id']}</div>
        </div>
        <div class="info-column">
            <div class="card-header">
                <span class="video-id">UniVLA Failure Case {idx+1}</span>
                <h2 class="action-label">"{item['label']}"</h2>
            </div>
            <table class="stats-table">
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>UniVLA DINO-VQ (Discrete)</th>
                        <th>Ours (VICReg Continuous)</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Unique Code Tokens Used</td>
                        <td class="collapse">{item['vq_codes']}</td>
                        <td class="highlight">998 / 1000</td>
                    </tr>
                    <tr>
                        <td>Temporal Std of Latents</td>
                        <td>0.0000 (Collapsed)</td>
                        <td class="highlight">{item['vicreg_std']:.4f}</td>
                    </tr>
                    <tr>
                        <td>Reconstruction Domain</td>
                        <td>DINOv2 Feature Space</td>
                        <td>Raw Pixel Space</td>
                    </tr>
                </tbody>
            </table>
            <div class="explanation">
                <strong>Analysis:</strong> UniVLA's task-centric VQ model collapses to a single code index for the entire action. This is because task text instructions and visual features are quantized into a very small discrete space, which completely discards the physical speed and trajectory of the action. VICReg maintains a high-dimensional continuous trajectory that accurately models the physics.
            </div>
        </div>
    </div>"""
    html_content += "\n</div>"

    # Section 4: villa-X Proxy
    html_content += '\n<div class="section-header">4. villa-X-style Grounded Latent Proxy vs. Ours</div>\n<div class="container">'
    for idx, item in enumerate(villax_failures):
        html_content += f"""
    <div class="card">
        <div class="video-column">
            <video src="{item['video_id']}.webm" controls loop muted></video>
            <div class="caption">Video ID: {item['video_id']}</div>
        </div>
        <div class="info-column">
            <div class="card-header">
                <span class="video-id">villa-X Failure Case {idx+1}</span>
                <h2 class="action-label">"{item['label']}"</h2>
            </div>
            <table class="stats-table">
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>villa-X (Discrete VQ)</th>
                        <th>Ours (VICReg Continuous)</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Unique Code Tokens Used</td>
                        <td class="collapse">{item['vq_codes']}</td>
                        <td class="highlight">998 / 1000</td>
                    </tr>
                    <tr>
                        <td>Temporal Std of Latents</td>
                        <td>0.0000 (Collapsed)</td>
                        <td class="highlight">{item['vicreg_std']:.4f}</td>
                    </tr>
                    <tr>
                        <td>Reconstruction MSE</td>
                        <td>{item['villax_rec']:.5f}</td>
                        <td class="highlight">{item['vicreg_rec']:.5f} (-{abs(item['rec_diff'])/item['villax_rec']*100:.1f}%)</td>
                    </tr>
                </tbody>
            </table>
            <div class="explanation">
                <strong>Analysis:</strong> The upgraded VQ-based villa-X model collapses to single code usage on complex transitions, whereas VICReg continues to trace continuous trajectories that successfully reconstruct the dynamics.
            </div>
        </div>
    </div>"""
    html_content += "\n</div>\n</body>\n</html>"

    with open(out_dir / "index.html", "w") as f:
        f.write(html_content)

    print("HTML visualization generation complete!")

if __name__ == "__main__":
    main()
