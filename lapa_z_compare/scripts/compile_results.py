import pandas as pd
import glob
import os
from pathlib import Path

def main():
    print("Compiling results from metrics.csv files under outputs/5k_v2...")
    
    # Files pattern
    files = glob.glob('lapa_z_compare/outputs/5k_v2/*/metrics.csv')
    
    # Mapping method names to display names
    display_names = {
        'lapa_nsvq': 'LAPA-style discrete VQ / NSVQ',
        'adaworld_bvae': 'AdaWorld-style continuous beta-VAE',
        'univla_dino_vq': 'UniVLA-style task-centric DINO/VQ',
        'villax_proxy': 'villa-X-style grounded latent proxy (discrete)',
        'vicreg': 'Ours: VICReg continuous'
    }
    
    records = []
    
    for f in files:
        method = Path(f).parent.name
        if method not in display_names:
            continue
            
        df = pd.read_csv(f)
        row = df.iloc[0].to_dict()
        row['method_key'] = method
        row['Method'] = display_names[method]
        records.append(row)
        
    if not records:
        print("No metrics.csv files found in outputs/5k_v2. Cannot compile results.")
        return

    # Optional: VLM task-progress metrics (TOPReward). Keyed by method.
    progress_path = Path('lapa_z_compare/outputs/5k_v2/_vlm_eval/progress_summary.csv')
    progress_by_method = {}
    if progress_path.exists():
        pdf_prog = pd.read_csv(progress_path)
        progress_by_method = {r['method']: r.to_dict() for _, r in pdf_prog.iterrows()}
        
    # Sort by a fixed order for consistency
    method_order = ['lapa_nsvq', 'adaworld_bvae', 'univla_dino_vq', 'villax_proxy', 'vicreg']
    records = sorted(records, key=lambda x: method_order.index(x['method_key']) if x['method_key'] in method_order else 99)
    
    # 1. Compile HTML Table
    html_rows = []
    for r in records:
        method_key = r['method_key']
        is_discrete = method_key in ['lapa_nsvq', 'univla_dino_vq', 'villax_proxy']
        
        # Format reconstruction loss
        recon_val = f"{r['avg_recon_loss']:.5f}"
        if method_key == 'univla_dino_vq':
            recon_val = f"{recon_val}*" # Feature space footnote
            
        # Format VQ metrics
        perplexity_val = f"{r['code_perplexity']:.2f}" if is_discrete else "N/A"
        dead_code_val = f"{r['dead_code_pct']:.2f}%" if is_discrete else "N/A"
        
        html_rows.append(f"""
            <tr>
                <td>{r['Method']}</td>
                <td>{recon_val}</td>
                <td style="font-weight: 500; color: #b7791f;">{r['zero_z_recon_gap']:.5f}</td>
                <td style="font-weight: 500; color: #2b6cb0;">{r['random_z_recon_gap']:.5f}</td>
                <td>{r['effective_rank']:.2f}</td>
                <td>{r['participation_ratio']:.2f}</td>
                <td>{r['corr_offdiag']:.6f}</td>
                <td>{perplexity_val}</td>
                <td>{dead_code_val}</td>
                <td style="font-weight: 600; color: #2d3748;">{r['linear_template_acc']:.4f}</td>
                <td style="font-weight: 600; color: #2d3748;">{r['linear_top5_acc']:.4f}</td>
                <td style="font-weight: 600; color: #2d3748;">{r['macro_f1']:.4f}</td>
                <td style="font-weight: 600; color: #1a202c;">{r['knn_same_template_rate']:.4f}</td>
            </tr>
        """)
        
    # Build the optional VLM task-progress section.
    progress_section = ""
    if progress_by_method:
        prog_rows = []
        for r in records:
            mk = r['method_key']
            p = progress_by_method.get(mk)
            if p is None:
                continue
            star = "&dagger;" if mk == 'univla_dino_vq' else ""
            prog_rows.append(f"""
            <tr>
                <td>{r['Method']}{star}</td>
                <td>{p['progress_true']:.3f}</td>
                <td style="font-weight:600; color:#b7791f;">{p['zero_z_progress_gap']:+.3f}</td>
                <td style="font-weight:600; color:#2b6cb0;">{p['random_z_progress_gap']:+.3f}</td>
                <td style="font-weight:600; color:#2f855a;">{p['wrong_template_contrast']:+.3f}</td>
                <td>{p['real_vs_decoded_gap']:.3f}</td>
                <td>{p['ordering_acc']:.3f}</td>
            </tr>
            """)
        progress_section = f"""
    <h2 style="margin-top:42px;">Task-Progress Evaluation (VLM / TOPReward)</h2>
    <div class="subtitle">Does the decoded future x&#770;<sub>t+H</sub>(z) read as task progress toward the STHV2 template? (Qwen3-VL-8B, n={int(list(progress_by_method.values())[0]['n'])} test pairs)</div>
    <div class="table-container">
        <table>
            <thead>
                <tr>
                    <th>Method</th>
                    <th>Progress(true z) &uarr;</th>
                    <th>Zero-Z Prog Gap &uarr;</th>
                    <th>Rand-Z Prog Gap &uarr;</th>
                    <th>Wrong-Tmpl Contrast &uarr;</th>
                    <th>Real&minus;Decoded Gap &darr;</th>
                    <th>Ordering Acc &uarr;</th>
                </tr>
            </thead>
            <tbody>
                {"".join(prog_rows)}
            </tbody>
        </table>
    </div>
    <div class="footnote">
        <strong>Task-Progress (VLM) notes:</strong>
        <ul>
            <li>Scorer = TOPReward (Chen et al.): probability the VLM assigns to the affirmative <i>True</i> token for "does the last frame show progress on the task?". Scorer validated on real STHV2 clips (median Value-Order Correlation &asymp; 0.81, 92% positive), confirming it tracks progress in-domain before scoring decoded frames.</li>
            <li><b>Zero-Z / Rand-Z / Wrong-Tmpl</b> hold the decoder fixed and vary only the latent or the instruction: positive values mean <i>z</i> adds task-relevant, sample-specific, action-matching information. These are the semantic analogues of the pixel-space zero-z / random-z gaps.</li>
            <li><b>Real&minus;Decoded Gap</b> is how much more the real future scores than the decoded future; large values indicate the 5k-tier decoders produce futures too low-fidelity for the VLM to read as progress.</li>
            <li>&dagger;UniVLA decodes DINOv2 features, not pixels; its scored frame is the nearest real future frame retrieved from the test set (excluding the sample's own future). Its absolute progress is therefore not directly comparable to the pixel decoders.</li>
            <li>These metrics evaluate the joint <i>z + decoder + VLM</i> pipeline, complementing (not replacing) the geometry and probe metrics above.</li>
        </ul>
    </div>
"""

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Latent Action Representation Quality Comparison</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            margin: 0;
            padding: 40px;
            display: flex;
            flex-direction: column;
            align-items: center;
            min-height: 100vh;
        }}
        .container {{
            max-width: 1300px;
            width: 100%;
            background-color: white;
            padding: 40px;
            border-radius: 16px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
        }}
        h2 {{
            color: #1f4e79;
            margin-top: 0;
            margin-bottom: 8px;
            font-weight: 700;
            font-size: 26px;
            text-align: center;
        }}
        .subtitle {{
            color: #4a5568;
            font-size: 15px;
            margin-bottom: 30px;
            text-align: center;
            font-weight: 500;
        }}
        .table-container {{
            overflow-x: auto;
            border-radius: 12px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            text-align: center;
        }}
        th {{
            background-color: #1f4e79;
            color: white;
            font-weight: 600;
            padding: 16px 14px;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            border-bottom: 2px solid #143554;
        }}
        td {{
            padding: 16px 14px;
            font-size: 13px;
            color: #2d3748;
            border-bottom: 1px solid #e2e8f0;
            transition: background-color 0.2s;
        }}
        tr td:first-child, tr th:first-child {{
            text-align: left;
            font-weight: 600;
            color: #1a202c;
            padding-left: 20px;
        }}
        tr:nth-child(even) td {{
            background-color: #f8fafc;
        }}
        tr:hover td {{
            background-color: #edf2f7;
        }}
        .footnote {{
            margin-top: 20px;
            font-size: 12px;
            color: #718096;
            line-height: 1.6;
        }}
        .footnote ul {{
            margin: 5px 0 0 20px;
            padding: 0;
        }}
    </style>
</head>
<body>

<div class="container">
    <h2>Latent Action Representation Quality Comparison</h2>
    <div class="subtitle">Continuous (Ours / VAE) vs. Discrete Codebook representations on STHV2 5K tier</div>
    
    <div class="table-container">
        <table>
            <thead>
                <tr>
                    <th>Method</th>
                    <th>Avg Recon Loss &darr;</th>
                    <th>Zero-Z Gap &uarr;</th>
                    <th>Rand-Z Gap &uarr;</th>
                    <th>Eff Rank &uarr;</th>
                    <th>Part Ratio &uarr;</th>
                    <th>Corr Off-diag &darr;</th>
                    <th>Perplexity &uarr;</th>
                    <th>Dead Code &darr;</th>
                    <th>Probe Top-1 &uarr;</th>
                    <th>Probe Top-5 &uarr;</th>
                    <th>Macro F1 &uarr;</th>
                    <th>kNN Rate &uarr;</th>
                </tr>
            </thead>
            <tbody>
                {"".join(html_rows)}
            </tbody>
        </table>
    </div>
    
    <div class="footnote">
        <strong>Notes and Disclaimers:</strong>
        <ul>
            <li>*Note: UniVLA reconstruction loss is computed in DINOv2 feature space (approx. range 1.5 - 2.5), whereas all other methods reconstruct raw pixels via MSE (approx. range 0.02 - 0.08). Do not compare reconstruction losses directly across different spaces.</li>
            <li>Zero-Z and Random-Z Reconstruction Gaps measure how much the latent <i>z</i> actively controls the output sequence prediction. Gaps are positive but small, indicating that visual context shortcuts and static frame conditioning dominate reconstruction across all models, with the action latent <i>z</i> playing a minor contributor role.</li>
            <li>Correlation Off-diagonal RMS provides a scale-invariant comparator for latent dimension decorrelation (unlike raw covariance off-diagonal RMS, which is scale-dependent).</li>
            <li>All models trained at the 5K tier achieve semantic probe accuracies close to chance-level on 174 imbalanced templates (linear probe accuracy ranges from 3.0% to 4.5%). When commitment loss is decoupled and trained correctly, discrete VQ models avoid posterior collapse (good code usage and perplexity).</li>
            <li>Perplexity and Dead Code Percentage are computed using true model codebook indices and are not applicable (N/A) for continuous latent representation methods.</li>
            <li>This evaluation is aimed at comparing model representations on a class project dataset tier, and display names are mapped to general architectural baselines rather than precise, full-scale paper reproductions.</li>
        </ul>
    </div>
    {progress_section}
</div>

</body>
</html>
"""
    
    html_path = Path('lapa_z_results_table.html')
    with open(html_path, 'w') as f_out:
        f_out.write(html_content)
    print(f"HTML results table compiled successfully to {html_path.resolve()}")
    
    # 2. Compile PDF Table (optional; requires matplotlib)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping PDF table (HTML written).")
        return

    columns = [
        "Method", "Recon\nLoss", "Zero-Z\nGap", "Rand-Z\nGap", 
        "Eff\nRank", "Part\nRatio", "Corr\nOffdiag", "Perplex", "Dead\nCode", 
        "Probe\nTop-1", "Probe\nTop-5", "Macro\nF1", "kNN\nRate"
    ]
    
    pdf_data = []
    for r in records:
        method_key = r['method_key']
        is_discrete = method_key in ['lapa_nsvq', 'univla_dino_vq', 'villax_proxy']
        
        recon_val = f"{r['avg_recon_loss']:.4f}"
        if method_key == 'univla_dino_vq':
            recon_val = f"{recon_val}*"
            
        perplexity_val = f"{r['code_perplexity']:.2f}" if is_discrete else "N/A"
        dead_code_val = f"{r['dead_code_pct']:.2f}%" if is_discrete else "N/A"
        
        pdf_data.append([
            r['Method'],
            recon_val,
            f"{r['zero_z_recon_gap']:.4f}",
            f"{r['random_z_recon_gap']:.4f}",
            f"{r['effective_rank']:.2f}",
            f"{r['participation_ratio']:.2f}",
            f"{r['corr_offdiag']:.5f}",
            perplexity_val,
            dead_code_val,
            f"{r['linear_template_acc']:.3f}",
            f"{r['linear_top5_acc']:.3f}",
            f"{r['macro_f1']:.3f}",
            f"{r['knn_same_template_rate']:.3f}"
        ])
        
    fig, ax = plt.subplots(figsize=(22, 5))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(
        cellText=pdf_data,
        colLabels=columns,
        loc='center',
        cellLoc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 2.2)
    
    header_bg = "#1f4e79"
    row_bg_even = "#ffffff"
    row_bg_odd = "#f8fafc"
    grid_color = "#e2e8f0"
    
    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor(header_bg)
        else:
            bg_color = row_bg_even if row_idx % 2 == 1 else row_bg_odd
            cell.set_facecolor(bg_color)
            if col_idx == 0:
                cell.set_text_props(ha='left', weight='bold')
                
        cell.set_edgecolor(grid_color)
        cell.set_linewidth(0.8)
        
    pdf_path = Path('/home/mohantyk/lapa_z_project/lapa_z_results_table.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', dpi=300)
    print(f"PDF results table compiled successfully to {pdf_path}")

if __name__ == '__main__':
    main()
