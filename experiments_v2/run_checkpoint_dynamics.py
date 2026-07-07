"""
MEMORIZATION ONSET across training checkpoints.

When, during training, does per-language memorization emerge -- and does it
emerge later / more slowly for low-resource or high-fertility languages?
Pythia ships 154 intermediate checkpoints (revisions ``step1000`` ...
``step143000``); OLMo-2 ships intermediate checkpoints too. We run the SAME
extraction attack on a FIXED evaluation set at each checkpoint and trace the
extraction curve per language.

Why this is clean: the evaluation (prefix, suffix) pairs are built ONCE and
reused across every checkpoint (the tokenizer is constant across revisions), so
differences are purely due to training progress, not the eval sample. This is
the cheapest way to get a *dynamics* figure and it uses open-data models (Pile)
where membership is known.

Prediction tied to the thesis: high-fertility languages show a later onset and
a shallower slope -- verbatim memorization requires reproducing more tokens, so
it takes more exposure to lock in.

Usage:
  python run_checkpoint_dynamics.py --model EleutherAI/pythia-1.4b \
      --lang finnish --n 1000 \
      --revisions step1000 step4000 step16000 step36000 step64000 step100000 step143000 \
      --out results/dynamics
"""
from __future__ import annotations
import argparse, csv, os
import torch
from tqdm import tqdm

from common import load_model, set_seed
from data import make_pairs
from run_extraction import generate_suffix
from metrics import token_exact_match, token_prefix_match_len
import yaml

DEFAULT_PYTHIA_REVS = ["step1000", "step4000", "step16000", "step36000",
                       "step64000", "step100000", "step143000"]


def step_of(rev: str) -> int:
    """Parse the training step. Handles Pythia (``step143000``) and OLMo
    (``stage1-step5000-tokens...``) by taking the number right after 'step';
    falls back to the first integer, else -1 (keeps input order)."""
    import re
    m = re.search(r"step[_-]?(\d+)", rev)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)", rev)
    return int(m.group(1)) if m else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--lang", required=True)
    ap.add_argument("--config", default="configs.yaml")
    ap.add_argument("--revisions", nargs="+", default=DEFAULT_PYTHIA_REVS)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--prefix_len", type=int, default=150)
    ap.add_argument("--suffix_len", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/dynamics")
    args = ap.parse_args()
    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    cfg = yaml.safe_load(open(args.config))
    member_src = cfg["languages"][args.lang]["member"]

    # Build the fixed eval set ONCE (tokenizer is constant across revisions).
    _, tok0 = load_model(args.model, revision=args.revisions[-1])
    pairs = list(make_pairs(tok0, member_src, args.n, is_member=1,
                            prefix_len=args.prefix_len, suffix_len=args.suffix_len))
    del tok0
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    rows = []
    for rev in args.revisions:
        model, tok = load_model(args.model, revision=rev)
        n_exact, pml_sum = 0, 0
        for i in tqdm(range(0, len(pairs), args.batch_size), desc=f"{args.lang} {rev}"):
            batch = pairs[i:i + args.batch_size]
            gen = generate_suffix(model, tok, [p.prefix_text for p in batch],
                                  args.suffix_len, "greedy")
            for p, g in zip(batch, gen):
                g = g.tolist()
                n_exact += token_exact_match(g, p.suffix_ids, args.suffix_len)
                pml_sum += token_prefix_match_len(g, p.suffix_ids, args.suffix_len)
        rows.append(dict(model=args.model.split("/")[-1], lang=args.lang,
                         revision=rev, step=step_of(rev), n=len(pairs),
                         exact_pct=round(100 * n_exact / len(pairs), 4),
                         mean_prefix_match=round(pml_sum / len(pairs), 3)))
        print("[ckpt]", rows[-1])
        del model, tok
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    tag = f"{args.model.split('/')[-1]}_{args.lang}"
    out_csv = os.path.join(args.out, f"dynamics_{tag}.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print(f"\nMEMORIZATION ONSET ({tag})")
    print("step,exact_pct,mean_prefix_match")
    for r in sorted(rows, key=lambda r: r["step"]):
        print(f"{r['step']},{r['exact_pct']},{r['mean_prefix_match']}")
    print(f"\n-> {out_csv}. Overlay several --lang runs to compare onset/slope "
          "across languages; correlate slope with tokenizer fertility.")


if __name__ == "__main__":
    main()
