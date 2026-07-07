"""
Membership Inference (reference-free suite) with a real non-member control.

For each of a member set and a temporally-held-out non-member set we compute:
  loss, zlib-ratio, Min-K% Prob, Min-K%++.
We then report AUROC and TPR@1%FPR per metric per language. Reference-free
metrics (Min-K%/Min-K%++) mean this arm ALSO works for InkubaLM, which the v1
PPL-ratio could not (mismatched tokenizer with Pythia).

This is what makes the "discoverable memorization" claim quantitative:
AUC >> 0.5 means the model's confidence separates seen from unseen text in
that language; we then correlate AUC with tokenizer fertility across languages.

Usage:
  python run_mia.py --model LumiOpen/Llama-Poro-2-8B-base --lang finnish \
      --n 2000 --out results/
"""
from __future__ import annotations
import argparse, csv, os
import torch
from tqdm import tqdm

from common import load_model, set_seed, token_logprobs
from data import make_pairs
from metrics import (min_k_percent, min_k_percent_pp, zlib_entropy,
                     tokenizer_fertility, auc, tpr_at_low_fpr)
import yaml


def score_text(model, tok, prefix, suffix):
    text = prefix + suffix
    tgt_lp, mu, sigma = token_logprobs(model, tok, text)
    loss = -float(tgt_lp.mean())
    return dict(
        neg_loss=-loss,                                   # higher => member
        zlib_ratio=zlib_entropy(text) / (loss + 1e-8),    # Carlini'21
        min_k=min_k_percent(tgt_lp, 0.2),                 # higher => member
        min_k_pp=min_k_percent_pp(tgt_lp, mu, sigma, 0.2),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lang", required=True)
    ap.add_argument("--config", default="configs.yaml")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    lang_cfg = cfg["languages"][args.lang]
    set_seed(args.seed)

    model, tok = load_model(args.model)
    os.makedirs(args.out, exist_ok=True)

    def collect(src, is_member):
        pairs = list(make_pairs(tok, src, args.n, is_member))
        out = []
        for p in tqdm(pairs, desc=f"{args.lang} member={is_member}"):
            s = score_text(model, tok, p.prefix_text, p.suffix_text)
            s.update(is_member=is_member, doc_id=p.doc_id)
            out.append(s)
        return out

    members = collect(lang_cfg["member"], 1)
    nonmembers = collect(lang_cfg["nonmember"], 0)

    tag = f"{args.model.split('/')[-1]}_{args.lang}"
    with open(os.path.join(args.out, f"mia_{tag}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(members[0].keys()))
        w.writeheader(); w.writerows(members + nonmembers)

    # fertility (mechanism variable) + per-metric AUC / TPR@1%FPR
    fert = tokenizer_fertility(
        tok, [p.prefix_text for p in make_pairs(tok, lang_cfg["member"], 200, 1)])
    summary = {"model": tag, "lang": args.lang, "fertility": round(fert, 3)}
    for metric in ["neg_loss", "zlib_ratio", "min_k", "min_k_pp"]:
        mv = [r[metric] for r in members]
        nv = [r[metric] for r in nonmembers]
        summary[f"auc_{metric}"] = round(auc(mv, nv), 4)
        summary[f"tpr1_{metric}"] = round(tpr_at_low_fpr(mv, nv, 0.01), 4)

    sfile = os.path.join(args.out, "mia_summary.csv")
    write_header = not os.path.exists(sfile)
    with open(sfile, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        if write_header:
            w.writeheader()
        w.writerow(summary)
    print("[done]", summary)


if __name__ == "__main__":
    main()
