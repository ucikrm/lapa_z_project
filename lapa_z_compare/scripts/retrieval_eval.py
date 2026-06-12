"""
Retrieval-based latent (z) evaluation.

Part A (this script, no VLM): for each method,
  - standardize z by TRAIN(bank) mean/std,
  - cosine kNN from test queries into the train bank,
  - compute same-template retrieval metrics (top-1, top-5 mean, template-MRR,
    random baseline), with a leakage check,
  - extract REAL frames needed for VLM progress scoring into a SHARED frame
    cache (so x_t / real-future / random-future / wrong-template-future are
    identical across all methods) plus per-method NN-future frames, and write a
    per-method manifest consumed by score_retrieval_vlm.py.

Frames saved are REAL video frames (never decoder output) -> isolates z quality.
"""
import argparse
import os
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def load_frame(video_path, frame_idx, max_side=None):
    """Return an RGB uint8 HxWx3 frame from a video at frame_idx (clamped)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        cap.release()
        return None
    fi = max(0, min(int(frame_idx), n - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if max_side is not None:
        h, w = rgb.shape[:2]
        s = max_side / max(h, w)
        if s < 1.0:
            rgb = cv2.resize(rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return rgb


def save_png(rgb, path):
    if rgb is None:
        return False
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return True


def standardize(z, mu, sd):
    return (z - mu) / sd


def l2norm(z):
    return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--bank", required=True, help="latents_train.npz")
    ap.add_argument("--query", required=True, help="latents_test.npz")
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--frame_cache", required=True, help="shared frame cache dir (method-independent frames)")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--buffer", type=int, default=50, help="neighbors kept for MRR")
    ap.add_argument("--num_score", type=int, default=120, help="#queries to extract frames for VLM scoring")
    ap.add_argument("--gap", type=int, default=12)
    ap.add_argument("--max_side", type=int, default=320)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np_rng = np.random.RandomState(args.seed)

    bank = np.load(args.bank, allow_pickle=True)
    query = np.load(args.query, allow_pickle=True)

    zb = bank["z"].astype(np.float64)
    zq = query["z"].astype(np.float64)
    b_tmpl = bank["template"]
    q_tmpl = query["template"]
    b_vid = bank["video_id"].astype(str)
    q_vid = query["video_id"].astype(str)
    b_t = bank["t"].astype(int)
    q_t = query["t"].astype(int)
    b_nf = bank["num_frames"].astype(int)
    q_nf = query["num_frames"].astype(int)
    q_label = query["label"]

    Nb, Nq = zb.shape[0], zq.shape[0]

    # ---- leakage check ----
    overlap = set(b_vid) & set(q_vid)
    leak = len(overlap)
    if leak:
        print(f"[WARN] {leak} overlapping video_ids between bank and query (possible leakage)")
    else:
        print("[OK] no video_id overlap between bank and query")

    # ---- standardize by bank mean/std, then L2 for cosine ----
    mu = zb.mean(0, keepdims=True)
    sd = zb.std(0, keepdims=True) + 1e-8
    zb_s = l2norm(standardize(zb, mu, sd))
    zq_s = l2norm(standardize(zq, mu, sd))

    K = min(args.buffer, Nb)
    # cosine similarity (vectors already standardized + L2-normalized)
    sim = zq_s @ zb_s.T  # (Nq, Nb)
    # top-K by descending similarity
    part = np.argpartition(-sim, kth=K - 1, axis=1)[:, :K]
    rows_ix = np.arange(Nq)[:, None]
    order_k = np.argsort(-sim[rows_ix, part], axis=1)
    idx = part[rows_ix, order_k]  # (Nq, K) bank indices, similarity-sorted
    dist = 1.0 - sim[rows_ix, idx]  # cosine distance

    # exclude any same-video bank hit (defensive), keep order
    def filtered_neighbors(qi):
        out = []
        for j in idx[qi]:
            if b_vid[j] == q_vid[qi]:
                continue
            out.append(int(j))
        return out

    topk = args.topk
    top1_hit, topk_rate, mrr = [], [], []
    rand_base = []
    # analytic random same-template prob per template
    tmpl_counts = pd.Series(b_tmpl).value_counts().to_dict()
    for qi in range(Nq):
        nbrs = filtered_neighbors(qi)
        if not nbrs:
            continue
        same = [b_tmpl[j] == q_tmpl[qi] for j in nbrs]
        top1_hit.append(1.0 if same[0] else 0.0)
        topk_rate.append(float(np.mean(same[:topk])))
        # MRR over full buffer
        r = next((rk + 1 for rk, s in enumerate(same) if s), None)
        mrr.append(1.0 / r if r else 0.0)
        rand_base.append(tmpl_counts.get(q_tmpl[qi], 0) / Nb)

    metrics = {
        "method": args.method,
        "n_query": Nq,
        "n_bank": Nb,
        "leakage_videos": leak,
        "top1_same_template": float(np.mean(top1_hit)),
        f"top{topk}_mean_same_template": float(np.mean(topk_rate)),
        "template_MRR": float(np.mean(mrr)),
        "random_same_template_baseline": float(np.mean(rand_base)),
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(out_dir / "retrieval_metrics.csv", index=False)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()})

    # save neighbors for grids
    nbr_idx = np.full((Nq, topk), -1, dtype=np.int64)
    for qi in range(Nq):
        nbrs = filtered_neighbors(qi)[:topk]
        nbr_idx[qi, : len(nbrs)] = nbrs
    np.savez_compressed(out_dir / "retrieval_neighbors.npz", neighbor_idx=nbr_idx, dist=dist[:, :topk])

    # ---- frame extraction for VLM scoring subset ----
    # video_id -> path maps
    tr_df = pd.read_csv(args.train_csv)
    te_df = pd.read_csv(args.test_csv)
    b_path = {str(r.video_id): r.video_path for r in tr_df.itertuples()}
    q_path = {str(r.video_id): r.video_path for r in te_df.itertuples()}

    cache = Path(args.frame_cache)

    def fut_idx(t, nf):
        return min(int(t) + args.gap, int(nf) - 1)

    # pick a deterministic query subset (shared across methods via same seed)
    order = list(range(Nq))
    rng.shuffle(order)
    subset = order[: args.num_score]

    rows = []
    for qi in subset:
        qd = cache / f"q{qi:04d}"
        xt_p = qd / "xt.png"
        real_p = qd / "real_future.png"
        rand_p = qd / "random_future.png"
        wrong_p = qd / "wrong_future.png"

        # method-independent shared frames: extract once
        if not xt_p.exists():
            save_png(load_frame(q_path[q_vid[qi]], q_t[qi], args.max_side), xt_p)
        if not real_p.exists():
            save_png(load_frame(q_path[q_vid[qi]], fut_idx(q_t[qi], q_nf[qi]), args.max_side), real_p)
        if not rand_p.exists():
            rj = np_rng.randint(0, Nb)
            save_png(load_frame(b_path[b_vid[rj]], fut_idx(b_t[rj], b_nf[rj]), args.max_side), rand_p)
        if not wrong_p.exists():
            # random bank entry with a DIFFERENT template
            wj = np_rng.randint(0, Nb)
            tries = 0
            while b_tmpl[wj] == q_tmpl[qi] and tries < 50:
                wj = np_rng.randint(0, Nb)
                tries += 1
            save_png(load_frame(b_path[b_vid[wj]], fut_idx(b_t[wj], b_nf[wj]), args.max_side), wrong_p)

        # per-method NN futures
        nbrs = filtered_neighbors(qi)[:topk]
        nn_paths = []
        for k, j in enumerate(nbrs):
            p = out_dir / "nn_frames" / f"q{qi:04d}_nn{k}.png"
            if not p.exists():
                save_png(load_frame(b_path[b_vid[j]], fut_idx(b_t[j], b_nf[j]), args.max_side), p)
            nn_paths.append(str(p))
        nn_paths += [""] * (topk - len(nn_paths))

        rows.append({
            "method": args.method,
            "q_idx": qi,
            "instruction": str(q_label[qi]),
            "template": str(q_tmpl[qi]),
            "xt": str(xt_p),
            "real_future": str(real_p),
            "random_future": str(rand_p),
            "wrong_future": str(wrong_p),
            **{f"nn{k}": nn_paths[k] for k in range(topk)},
        })

    man = pd.DataFrame(rows)
    man.to_csv(out_dir / "vlm_manifest.csv", index=False)
    print(f"Wrote manifest with {len(man)} queries -> {out_dir/'vlm_manifest.csv'}")


if __name__ == "__main__":
    main()
