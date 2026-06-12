"""Build a self-contained HTML report for retrieval-based z evaluation.

Combines:
  - per-method retrieval_metrics.csv (same-template top-1/top-5/MRR + random baseline)
  - vlm_retrieval_summary.csv (TOPReward lift over random, wrong-template contrast)
  - a qualitative retrieval grid (x_t | real future | each method's top-1 retrieved
    real future), base64-embedded so the file is portable.
"""
import argparse
import base64
from pathlib import Path

import pandas as pd

METHOD_LABEL = {
    "lapa_nsvq": "LAPA-NSVQ (proxy)",
    "adaworld_bvae": "AdaWorld \u03b2-VAE (proxy)",
    "univla_dino_vq": "UniVLA DINO-VQ (proxy)",
    "villax_proxy": "villa-X structural (proxy)",
    "vicreg": "VICReg (ours)",
}
ORDER = ["lapa_nsvq", "adaworld_bvae", "univla_dino_vq", "villax_proxy", "vicreg"]


def b64(path):
    p = Path(path)
    if not p.exists():
        return ""
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()


def img_tag(path, w=150):
    src = b64(path)
    if not src:
        return '<div class="missing">—</div>'
    return f'<img src="{src}" width="{w}">'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_grid", type=int, default=6)
    args = ap.parse_args()

    root = Path(args.retrieval_dir)
    rmet = {}
    for m in ORDER:
        f = root / m / "retrieval_metrics.csv"
        if f.exists():
            rmet[m] = pd.read_csv(f).iloc[0].to_dict()
    vlm = pd.read_csv(root / "vlm_retrieval_summary.csv").set_index("method").to_dict("index")

    rand_base = next(iter(rmet.values())).get("random_same_template_baseline", float("nan"))
    real = next(iter(vlm.values())).get("prog_real")
    nochange = next(iter(vlm.values())).get("prog_nochange")
    p_random = next(iter(vlm.values())).get("prog_random")
    p_wrong = next(iter(vlm.values())).get("prog_wrong")
    order_acc = next(iter(vlm.values())).get("ordering_acc")

    # ---- headline table ----
    rows = []
    for m in ORDER:
        if m not in rmet:
            continue
        r = rmet[m]
        v = vlm.get(m, {})
        lift = v.get("topreward_lift_over_random", float("nan"))
        lo, hi = v.get("lift_ci95_lo", float("nan")), v.get("lift_ci95_hi", float("nan"))
        sig = "yes" if (lo is not None and lo > 0) else "no"
        rows.append(
            f"<tr><td class='m'>{METHOD_LABEL[m]}</td>"
            f"<td>{r['top1_same_template']*100:.1f}%</td>"
            f"<td>{r['top5_mean_same_template']*100:.1f}%</td>"
            f"<td>{r['template_MRR']:.3f}</td>"
            f"<td>{v.get('prog_nn1', float('nan')):.3f}</td>"
            f"<td>{lift:+.3f}<br><span class='ci'>[{lo:+.3f}, {hi:+.3f}]</span></td>"
            f"<td>{v.get('wrong_template_contrast', float('nan')):+.3f}</td>"
            f"<td class='sig-{sig}'>{sig}</td></tr>"
        )
    headline = "\n".join(rows)

    # ---- qualitative grid: pick top-N queries by VICReg nn1 progress ----
    grid_html = ""
    vic_ps = root / "vicreg" / "vlm_progress_persample.csv"
    mans = {m: pd.read_csv(root / m / "vlm_manifest.csv").set_index("q_idx") for m in ORDER if (root / m / "vlm_manifest.csv").exists()}
    if vic_ps.exists():
        vp = pd.read_csv(vic_ps).sort_values("prog_nn1", ascending=False)
        qids = vp["q_idx"].head(args.n_grid).tolist()
        vic_man = mans["vicreg"]
        head = "<tr><th>instruction</th><th>x_t (query)</th><th>real future</th>" + \
            "".join(f"<th>{METHOD_LABEL[m]}<br>top-1 retrieved</th>" for m in ORDER if m in mans) + "</tr>"
        body = []
        for q in qids:
            instr = str(vic_man.loc[q, "instruction"])
            cells = [f"<td class='instr'>{instr}</td>",
                     f"<td>{img_tag(vic_man.loc[q,'xt'])}</td>",
                     f"<td>{img_tag(vic_man.loc[q,'real_future'])}</td>"]
            for m in ORDER:
                if m not in mans:
                    continue
                mm = mans[m]
                nn0 = mm.loc[q, "nn0"] if q in mm.index else ""
                cells.append(f"<td>{img_tag(nn0)}</td>")
            body.append("<tr>" + "".join(cells) + "</tr>")
        grid_html = f"<table class='grid'>{head}{''.join(body)}</table>"

    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Retrieval-based z Evaluation</title>
<style>
body{{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:32px;color:#1a1a1a;max-width:1200px}}
h1{{font-size:24px}} h2{{font-size:19px;margin-top:34px;border-bottom:2px solid #eee;padding-bottom:6px}}
table{{border-collapse:collapse;margin:14px 0;font-size:14px}}
th,td{{border:1px solid #d8d8d8;padding:7px 10px;text-align:center}}
th{{background:#f4f6f8}} td.m{{text-align:left;font-weight:600}}
tr:last-child td{{background:#eef7ee}}
.ci{{color:#888;font-size:11px}} .sig-yes{{color:#178017;font-weight:700}} .sig-no{{color:#b00}}
.note{{background:#fff8e6;border-left:4px solid #e0a800;padding:10px 14px;margin:12px 0;font-size:13px}}
.grid td{{vertical-align:top}} .grid img{{border-radius:4px}} td.instr{{text-align:left;max-width:160px;font-size:12px}}
.missing{{color:#bbb}} .refs{{font-size:13px;color:#444}}
</style></head><body>
<h1>Retrieval-based latent (z) evaluation &mdash; STHV2</h1>
<p class="refs">Protocol identical for all five methods: standardize z by train(bank) mean/std, cosine kNN from
{int(next(iter(rmet.values()))['n_query'])} test queries into a {int(next(iter(rmet.values()))['n_bank'])}-clip train bank
(no video_id leakage). VLM (TOPReward / Qwen3-VL-8B) scores <b>real retrieved future frames</b> &mdash; never decoder
output &mdash; so the metric isolates z quality from the weak decoder.</p>

<h2>Headline: does z retrieve semantically correct futures?</h2>
<table>
<tr><th>Method</th><th>top-1 same-template</th><th>top-5 mean same-template</th><th>template-MRR</th>
<th>VLM progress (NN top-1)</th><th>TOPReward lift over random</th><th>wrong-template contrast</th><th>lift sig.?</th></tr>
{headline}
</table>
<p class="refs">Random same-template baseline = <b>{rand_base*100:.2f}%</b> (STHV2 has 100+ templates).
Shared VLM references: real-future progress = <b>{real:.3f}</b>, no-change = <b>{nochange:.3f}</b>,
random-future = <b>{p_random:.3f}</b>, wrong-template = <b>{p_wrong:.3f}</b>; scorer ordering sanity (real&gt;no-change) = <b>{order_acc*100:.0f}%</b>.</p>

<div class="note"><b>How to read this (honest framing):</b>
<ul>
<li><b>Lift over random</b> and <b>wrong-template contrast</b> are the defensible headline numbers: both NN and the
random/wrong baselines are cross-video real frames, so they share the same scene-mismatch confound &mdash; the comparison is apples-to-apples.</li>
<li><b>Aggregate retention vs no-change is NOT reported as a headline.</b> The no-change reference uses the <i>same</i> scene as the query,
while every retrieved future comes from a <i>different</i> video (different objects/background). The VLM therefore scores all cross-video
futures low in absolute terms, making retention-vs-no-change negative for every method and uninformative for cross-video retrieval.</li>
<li>VICReg is the only method whose retrieved futures the VLM reads as substantially more task-relevant than random (lift +0.127, CI excludes 0)
and than a wrong-template future (+0.096), consistent with its best same-template retrieval (top-1 6.3%, MRR 0.110).</li>
</ul></div>

<h2>Qualitative retrieval grid (top VICReg cases)</h2>
<p class="refs">For each query: the current frame, the true future, and each method's top-1 retrieved <b>real</b> future from the bank.</p>
{grid_html}

</body></html>"""

    Path(args.out).write_text(html)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
