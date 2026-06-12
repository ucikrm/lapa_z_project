"""
Phase 3 (decode step) — runs in the `llm312` training env.

For a subset of the test split, decode the predicted future frame x_hat_{t+H}
with the TRUE latent z, a ZERO latent (z=0), and a RANDOM latent (z rolled across
the batch). Save RGB frames + a manifest for the VLM scoring step.

Pixel-decoding methods (lapa_nsvq, adaworld_bvae, villax_proxy, vicreg) save x_hat
directly. UniVLA decodes DINOv2 features, so we map each predicted feature map to the
nearest *real* future frame in the subset (excluding the sample's own future) and save
that retrieved frame instead.

Outputs under <out_root>/:
  shared/<idx>_xt.png, shared/<idx>_real.png         (method-independent)
  <method>/<idx>_true.png, _zero.png, _random.png
  <method>/manifest.csv
"""
import argparse
import csv
from pathlib import Path
import sys

import numpy as np
import torch
from torchvision.utils import save_image

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from lapa_z_compare.src.data.sthv2_pair_dataset import SthV2PairDataset
from lapa_z_compare.src.models.lapa_nsvq import LapaNSVQModel
from lapa_z_compare.src.models.adaworld_bvae import AdaWorldBVAEModel
from lapa_z_compare.src.models.univla_dino_vq import UniVLADinoVQModel
from lapa_z_compare.src.models.villax_proxy import VillaXProxyModel
from lapa_z_compare.src.models.vicreg_continuous import VicregContinuousModel


def build_model(method, train_args):
    latent_dim = train_args["latent_dim"]
    if method == "lapa_nsvq":
        return LapaNSVQModel(latent_dim=latent_dim, num_codes=train_args.get("num_codes", 64),
                             beta=train_args.get("vq_beta", train_args.get("beta", 0.25)))
    if method == "adaworld_bvae":
        return AdaWorldBVAEModel(latent_dim=latent_dim, beta=train_args.get("beta", 2e-4))
    if method == "univla_dino_vq":
        return UniVLADinoVQModel(latent_dim=latent_dim, num_codes=train_args.get("num_codes", 64),
                                 beta=train_args.get("vq_beta", train_args.get("beta", 0.25)))
    if method == "villax_proxy":
        return VillaXProxyModel(latent_dim=latent_dim, lambda_s=train_args.get("lambda_s", 1.0),
                                beta=train_args.get("vq_beta", 0.25), num_codes=train_args.get("num_codes", 64))
    if method == "vicreg":
        return VicregContinuousModel(latent_dim=latent_dim, lambda_var=train_args.get("lambda_var", 1.0),
                                     lambda_cov=train_args.get("lambda_cov", 0.05))
    raise ValueError(f"Unknown method {method}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--subset", type=int, default=300)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    train_args = ckpt["args"]
    method = train_args["method"]
    assert method == args.method, f"ckpt method {method} != {args.method}"

    model = build_model(method, train_args)
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()

    is_univla = method == "univla_dino_vq"
    dataset = SthV2PairDataset(args.test_csv, image_size=train_args["image_size"],
                               gap=train_args["gap"], split="test", compute_flow=False)
    n = min(args.subset, len(dataset))

    out_root = Path(args.out_root)
    shared_dir = out_root / "shared"
    method_dir = out_root / method
    shared_dir.mkdir(parents=True, exist_ok=True)
    method_dir.mkdir(parents=True, exist_ok=True)

    # Load all subset samples once (deterministic test-split center sampling).
    xts, x2s, templates, vids = [], [], [], []
    for i in range(n):
        s = dataset[i]
        xts.append(s["x1"]); x2s.append(s["x2"])
        templates.append(s["template"]); vids.append(s["video_id"])
    xt_batch = torch.stack(xts)
    x2_batch = torch.stack(x2s)

    # Precompute real-future DINO feature bank for UniVLA NN-retrieval.
    bank = None
    if is_univla:
        feats = []
        with torch.no_grad():
            for i in range(0, n, 32):
                xb = x2_batch[i:i + 32].to(device)
                f = model.get_dino_features(xb).mean(dim=(2, 3))  # [b,768]
                feats.append(f.cpu())
        bank = torch.cat(feats, dim=0)  # [n,768]
        bank = bank / (bank.norm(dim=1, keepdim=True) + 1e-8)

    # Save shared frames (method-independent).
    for i in range(n):
        xt_p = shared_dir / f"{i:05d}_xt.png"
        real_p = shared_dir / f"{i:05d}_real.png"
        if not xt_p.exists():
            save_image(xt_batch[i], xt_p)
        if not real_p.exists():
            save_image(x2_batch[i], real_p)

    rows = []
    bs = 32
    with torch.no_grad():
        for i in range(0, n, bs):
            xt = xt_batch[i:i + bs].to(device)
            x2 = x2_batch[i:i + bs].to(device)
            lbls = templates[i:i + bs]

            if is_univla:
                out = model(xt, x2, label=lbls)
                z = out["z"]
                feat_true = out["x_hat"].mean(dim=(2, 3))
                z_zero = torch.zeros_like(z)
                z_rand = torch.roll(z, shifts=1, dims=0)
                feat_zero = model.decode(xt, z_zero, label=lbls).mean(dim=(2, 3))
                feat_rand = model.decode(xt, z_rand, label=lbls).mean(dim=(2, 3))
                # NN-retrieve nearest real frame for each predicted feature (exclude self).
                for j in range(xt.shape[0]):
                    gi = i + j
                    for tag, feat in (("true", feat_true), ("zero", feat_zero), ("random", feat_rand)):
                        q = feat[j:j + 1].cpu()
                        q = q / (q.norm(dim=1, keepdim=True) + 1e-8)
                        sims = (q @ bank.t()).squeeze(0)
                        sims[gi] = -1e9  # exclude own future
                        nn_idx = int(sims.argmax().item())
                        save_image(x2_batch[nn_idx], method_dir / f"{gi:05d}_{tag}.png")
            else:
                out = model(xt, x2)
                z = out["z"]
                z_zero = torch.zeros_like(z)
                z_rand = torch.roll(z, shifts=1, dims=0)
                xhat_true = out["x_hat"].clamp(0, 1)
                xhat_zero = model.decode(xt, z_zero).clamp(0, 1)
                xhat_rand = model.decode(xt, z_rand).clamp(0, 1)
                for j in range(xt.shape[0]):
                    gi = i + j
                    save_image(xhat_true[j], method_dir / f"{gi:05d}_true.png")
                    save_image(xhat_zero[j], method_dir / f"{gi:05d}_zero.png")
                    save_image(xhat_rand[j], method_dir / f"{gi:05d}_random.png")

    # Wrong-template control: roll the template list by a fixed offset.
    wrong = templates[7:] + templates[:7]
    for i in range(n):
        rows.append({
            "idx": i,
            "video_id": vids[i],
            "template": templates[i],
            "wrong_template": wrong[i],
            "xt": str((shared_dir / f"{i:05d}_xt.png").resolve()),
            "real": str((shared_dir / f"{i:05d}_real.png").resolve()),
            "true": str((method_dir / f"{i:05d}_true.png").resolve()),
            "zero": str((method_dir / f"{i:05d}_zero.png").resolve()),
            "random": str((method_dir / f"{i:05d}_random.png").resolve()),
        })

    manifest = method_dir / "manifest.csv"
    with open(manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[{method}] decoded {n} samples -> {manifest}")


if __name__ == "__main__":
    main()
