import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from collections import Counter

def compute_rank_and_pr(z):
    """
    Computes both effective rank and participation ratio of a latent matrix Z.
    """
    # Center the matrix
    z_centered = z - np.mean(z, axis=0)
    
    # Compute singular values
    _, s, _ = np.linalg.svd(z_centered, full_matrices=False)
    
    # Handle zero singular values
    s_sum = np.sum(s)
    if s_sum < 1e-10:
        eff_rank = 1.0
    else:
        p = s / s_sum
        p = np.where(p > 0, p, 1e-12)
        entropy = -np.sum(p * np.log(p))
        eff_rank = np.exp(entropy)
        
    s_sq = s ** 2
    s_sq_sum = np.sum(s_sq)
    if s_sq_sum < 1e-10:
        pr = 1.0
    else:
        pr = (s_sq_sum ** 2) / np.sum(s_sq ** 2)
        
    return eff_rank, pr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, required=True, help="Path to exported latents_test.npz")
    parser.add_argument("--out_csv", type=str, required=True, help="Path to save output metrics CSV")
    parser.add_argument("--num_codes", type=int, default=64, help="Total number of codes in the VQ codebook")
    args = parser.parse_args()

    # Load data
    data = np.load(args.npz)
    z = data["z"]
    video_id = data["video_id"]
    label = data["label"]
    template = data["template"]
    recon_loss = data["recon_loss"]

    # Load zero-z and random-z recon losses if available, else default to recon_loss (zero gap)
    recon_loss_zero = data["recon_loss_zero"] if "recon_loss_zero" in data.files else recon_loss
    recon_loss_random = data["recon_loss_random"] if "recon_loss_random" in data.files else recon_loss
    indices = data["indices"] if "indices" in data.files else -np.ones(len(z), dtype=np.int32)

    print(f"Loaded latents from {args.npz}")
    print(f"Latents shape: {z.shape}")

    # 1. Global Geometric Metrics
    effective_rank, participation_ratio = compute_rank_and_pr(z)
    
    stds = np.std(z, axis=0)
    std_mean = np.mean(stds)
    std_min = np.min(stds)
    std_max = np.max(stds)
    
    cov = np.cov(z, rowvar=False)
    if z.shape[1] > 1:
        off_diag = cov - np.diag(np.diag(cov))
        cov_offdiag = np.sqrt(np.mean(off_diag ** 2))
        
        # Scale-invariant correlation matrix
        denom = np.outer(stds, stds) + 1e-8
        corr = cov / denom
        off_corr = corr - np.diag(np.diag(corr))
        corr_offdiag = np.sqrt(np.mean(off_corr ** 2))
    else:
        cov_offdiag = 0.0
        corr_offdiag = 0.0

    avg_recon_loss = np.mean(recon_loss)
    avg_recon_loss_zero = np.mean(recon_loss_zero)
    avg_recon_loss_random = np.mean(recon_loss_random)
    
    # Compute gaps
    zero_z_recon_gap = avg_recon_loss_zero - avg_recon_loss
    random_z_recon_gap = avg_recon_loss_random - avg_recon_loss

    print("\n--- Geometric Metrics ---")
    print(f"Avg Recon Loss: {avg_recon_loss:.6f}")
    print(f"Avg Recon Loss (Z=0): {avg_recon_loss_zero:.6f}")
    print(f"Avg Recon Loss (Z=rand): {avg_recon_loss_random:.6f}")
    print(f"Zero-Z Recon Gap: {zero_z_recon_gap:.6f}")
    print(f"Random-Z Recon Gap: {random_z_recon_gap:.6f}")
    print(f"Effective Rank: {effective_rank:.4f} / {z.shape[1]}")
    print(f"Participation Ratio: {participation_ratio:.4f}")
    print(f"Std Mean: {std_mean:.6f}")
    print(f"Std Min:  {std_min:.6f}")
    print(f"Std Max:  {std_max:.6f}")
    print(f"Covariance Off-diagonal RMS: {cov_offdiag:.6f}")
    print(f"Correlation Off-diagonal RMS: {corr_offdiag:.6f}")

    # 2. VQ Specific Metrics (if applicable)
    is_discrete = np.any(indices >= 0)
    if is_discrete:
        # Determine codebook size dynamically if indices exceed num_codes
        num_codes = args.num_codes
        if indices.max() + 1 > num_codes:
            num_codes = int(indices.max() + 1)
            
        used_indices = indices[indices >= 0]
        unique_indices = np.unique(used_indices)
        counts = Counter(used_indices)
        p = np.array([counts.get(i, 0) for i in range(num_codes)]) / len(used_indices)
        p_used = p[p > 0]
        true_entropy = -np.sum(p_used * np.log(p_used))
        code_perplexity = np.exp(true_entropy)
        dead_code_pct = (num_codes - len(unique_indices)) / num_codes * 100
        
        # Estimate code index entropy based on unique vectors in z
        unique_vectors = np.unique(z, axis=0)
        num_unique_vectors = len(unique_vectors)
        vector_to_idx = {tuple(v): idx for idx, v in enumerate(unique_vectors)}
        est_indices = [vector_to_idx[tuple(v)] for v in z]
        counts_est = Counter(est_indices)
        p_est = np.array(list(counts_est.values())) / len(z)
        code_entropy = -np.sum(p_est * np.log(p_est))
    else:
        code_perplexity = -1.0
        dead_code_pct = -1.0
        num_unique_vectors = -1
        code_entropy = -1.0

    print("\n--- Codebook / Vector Discretization Metrics ---")
    print(f"Is Discrete: {is_discrete}")
    if is_discrete:
        print(f"Unique Latent Vectors: {num_unique_vectors}")
        print(f"Estimated Code Entropy: {code_entropy:.4f}")
        print(f"Code Perplexity: {code_perplexity:.4f}")
        print(f"Dead Code Percentage: {dead_code_pct:.2f}%")

    # 3. Label & Template Probes (Semantic Metrics)
    unique_templates = np.unique(template)
    template_to_idx = {t: idx for idx, t in enumerate(unique_templates)}
    y = np.array([template_to_idx[t] for t in template])

    linear_acc = 0.0
    linear_top5_acc = 0.0
    macro_f1 = 0.0
    
    if len(np.unique(y)) > 1 and len(z) >= 10:
        try:
            z_train, z_test, y_train, y_test = train_test_split(z, y, test_size=0.2, random_state=42, stratify=y)
        except Exception:
            z_train, z_test, y_train, y_test = train_test_split(z, y, test_size=0.2, random_state=42)
            
        try:
            clf = LogisticRegression(max_iter=1000)
            clf.fit(z_train, y_train)
            linear_acc = clf.score(z_test, y_test)
            
            y_pred = clf.predict(z_test)
            macro_f1 = f1_score(y_test, y_pred, average="macro")
            
            y_prob = clf.predict_proba(z_test)
            class_to_col = {c: i for i, c in enumerate(clf.classes_)}
            top5_correct = 0
            for i in range(len(y_test)):
                true_cls = y_test[i]
                if true_cls in class_to_col:
                    col = class_to_col[true_cls]
                    top5_cols = np.argsort(y_prob[i])[-5:]
                    if col in top5_cols:
                        top5_correct += 1
            linear_top5_acc = top5_correct / len(y_test)
        except Exception as e:
            print(f"Linear probe error: {e}")

    # 4. kNN Retrieval Metric (Same-template Rate)
    knn_same_rate = 0.0
    if len(z) >= 5:
        nbrs = NearestNeighbors(n_neighbors=6, algorithm='auto').fit(z)
        distances, indices_knn = nbrs.kneighbors(z)
        
        same_count = 0
        total_count = 0
        for i in range(len(z)):
            query_template = template[i]
            neighbor_indices = indices_knn[i][1:]
            for n_idx in neighbor_indices:
                if template[n_idx] == query_template:
                    same_count += 1
                total_count += 1
        knn_same_rate = same_count / max(total_count, 1)

    print("\n--- Action Semantics Probes ---")
    print(f"Linear Template Probe Accuracy: {linear_acc:.4f}")
    print(f"Linear Top-5 Probe Accuracy: {linear_top5_acc:.4f}")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    print(f"kNN Same-Template Retrieval Rate: {knn_same_rate:.4f}")

    # Save to CSV
    metrics = {
        "avg_recon_loss": avg_recon_loss,
        "avg_recon_loss_zero": avg_recon_loss_zero,
        "avg_recon_loss_random": avg_recon_loss_random,
        "zero_z_recon_gap": zero_z_recon_gap,
        "random_z_recon_gap": random_z_recon_gap,
        "effective_rank": effective_rank,
        "participation_ratio": participation_ratio,
        "std_mean": std_mean,
        "std_min": std_min,
        "std_max": std_max,
        "cov_offdiag": cov_offdiag,
        "corr_offdiag": corr_offdiag,
        "num_unique_vectors": num_unique_vectors,
        "code_entropy": code_entropy,
        "code_perplexity": code_perplexity,
        "dead_code_pct": dead_code_pct,
        "linear_template_acc": linear_acc,
        "linear_top5_acc": linear_top5_acc,
        "macro_f1": macro_f1,
        "knn_same_template_rate": knn_same_rate
    }
    
    df = pd.DataFrame([metrics])
    df.to_csv(args.out_csv, index=False)
    print(f"\nSaved all metrics to {args.out_csv}")

if __name__ == "__main__":
    main()
