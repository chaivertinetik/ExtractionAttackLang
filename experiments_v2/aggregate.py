"""
Turn raw per-sample CSVs into publication tables with bootstrap 95% CIs.

The #1 reviewer complaint about v1-style tables is "single run, no error bars,
magic thresholds." This produces:
  * extraction rate (%) with bootstrap CI per (model, language, decoding)
  * mean prefix-match length (approximate memorization) with CI
  * a LaTeX table you can paste into the paper

Usage:
  python aggregate.py --glob 'results/extraction_*.csv' --out tables/
  python aggregate.py --mia results/mia_summary.csv
"""
from __future__ import annotations
import argparse, csv, glob, os, re
import numpy as np


def bootstrap_ci(x, stat=np.mean, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    if len(x) == 0:
        return (float("nan"),) * 3
    boot = [stat(rng.choice(x, len(x), replace=True)) for _ in range(n)]
    return float(stat(x)), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def parse_tag(path):
    m = re.search(r"extraction_(.+?)_(\w+)_(greedy|topk)\.csv", os.path.basename(path))
    return (m.group(1), m.group(2), m.group(3)) if m else (os.path.basename(path), "?", "?")


def extraction_table(paths):
    rows = []
    for p in paths:
        data = load(p)
        model, lang, dec = parse_tag(p)
        ex = [int(r["is_exact"]) for r in data]
        pml = [int(r["prefix_match_len"]) for r in data]
        e_m, e_lo, e_hi = bootstrap_ci(ex)
        p_m, p_lo, p_hi = bootstrap_ci(pml)
        rows.append(dict(model=model, lang=lang, decoding=dec, n=len(data),
                         exact_pct=100 * e_m, exact_lo=100 * e_lo, exact_hi=100 * e_hi,
                         mean_prefix_match=p_m, pml_lo=p_lo, pml_hi=p_hi))
    return rows


def print_table(rows):
    print(f"{'model':30} {'lang':9} {'dec':6} {'n':>6} {'exact% [95% CI]':>24} {'prefixlen [CI]':>20}")
    for r in sorted(rows, key=lambda r: (r["model"], r["lang"])):
        print(f"{r['model'][:30]:30} {r['lang']:9} {r['decoding']:6} {r['n']:6d} "
              f"{r['exact_pct']:6.3f} [{r['exact_lo']:.3f},{r['exact_hi']:.3f}]   "
              f"{r['mean_prefix_match']:5.2f} [{r['pml_lo']:.1f},{r['pml_hi']:.1f}]")


def latex(rows, out):
    with open(out, "w") as f:
        f.write("\\begin{tabular}{llrrr}\n\\toprule\n")
        f.write("Model & Lang & $N$ & Exact (\\%) & Prefix-match \\\\\n\\midrule\n")
        for r in sorted(rows, key=lambda r: (r["model"], r["lang"])):
            f.write(f"{r['model']} & {r['lang']} & {r['n']} & "
                    f"{r['exact_pct']:.3f}\\,\\tiny[{r['exact_lo']:.3f},{r['exact_hi']:.3f}] & "
                    f"{r['mean_prefix_match']:.2f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="results/extraction_*.csv")
    ap.add_argument("--mia", default=None)
    ap.add_argument("--out", default="tables")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.mia:
        for r in load(args.mia):
            print(r)
        return

    rows = extraction_table(sorted(glob.glob(args.glob)))
    print_table(rows)
    latex(rows, os.path.join(args.out, "extraction.tex"))


if __name__ == "__main__":
    main()
