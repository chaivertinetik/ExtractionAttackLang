"""
GRAND-UNIFIED experiment: memorization as a 2-D causal surface over
    (duplication count)  x  (tokenizer fertility).

The multilingual-memorization literature has two separate mechanistic knobs:
  * DUPLICATION  -- Carlini'22 / Lee'22: memorization grows log-linearly with
    how many times a string is repeated in training.
  * FERTILITY    -- our thesis: how many subword tokens a string fragments into
    (agglutinative / poorly-covered languages fragment more -> less verbatim
    memorization at matched content).

No prior work puts them on the same axes. This script does, causally, by
sweeping BOTH under a controlled continued-pretrain of a small base model:

  fertility axis   : add K native subword tokens to the tokenizer, K in a sweep
                     {0, 1k, 5k, 20k}. More added tokens -> LOWER fertility.
  duplication axis : inject each canary at reps in {1,2,4,8,16,32,64,128}.

Output: extraction_rate(fertility_level, reps). The predicted surface:
  * monotone up in reps (duplication helps memorization), AND
  * monotone up as fertility DROPS (lower fertility -> more memorization),
  * with an interaction: fertility flattens the duplication slope for
    high-fertility (poorly-tokenized) languages -> a mechanistic explanation of
    why low-resource / morphologically rich languages resist extraction.

This one figure subsumes the canary experiment (a fertility slice) and the
vocab-intervention experiment (a 2-point fertility cut) and is the paper's
headline causal result.

Compute: trains ONE small model per fertility level (default 4). Use pythia-410m
on one A100, or a 160m for a fast first pass. Reuses machinery from
run_canary.py and run_vocab_intervention.py.

Usage:
  python run_fertility_sweep.py --base_model EleutherAI/pythia-410m \
      --corpus_hf HuggingFaceFW/fineweb-2 --corpus_config swh_Latn --lang swahili \
      --add_tokens_sweep 0 1000 5000 20000 \
      --reps 1 2 4 8 16 32 64 128 --out results/surface_swahili
"""
from __future__ import annotations
import argparse, copy, csv, json, os
import torch
from transformers import AutoTokenizer

from common import set_seed
from metrics import tokenizer_fertility
from run_canary import make_canaries, measure
from run_vocab_intervention import expand_tokenizer, train_condition


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="EleutherAI/pythia-410m")
    ap.add_argument("--corpus_hf", default="HuggingFaceFW/fineweb-2")
    ap.add_argument("--corpus_config", default="swh_Latn")
    ap.add_argument("--corpus_docs", type=int, default=20000)
    ap.add_argument("--lang", default="swahili")
    ap.add_argument("--add_tokens_sweep", type=int, nargs="+", default=[0, 1000, 5000, 20000])
    ap.add_argument("--reps", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128])
    ap.add_argument("--n_per_cell", type=int, default=25)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/surface")
    args = ap.parse_args()
    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    from datasets import load_dataset
    stream = load_dataset(args.corpus_hf, args.corpus_config, split="train", streaming=True)
    base_texts = []
    for row in stream:
        base_texts.append(row["text"])
        if len(base_texts) >= args.corpus_docs:
            break

    # identical canary CONTENT reused at every fertility level -> only
    # tokenization changes across the sweep.
    canaries = make_canaries([args.lang], args.reps, args.n_per_cell, args.seed)
    json.dump(canaries, open(os.path.join(args.out, "canaries.json"), "w"))

    base_tok = AutoTokenizer.from_pretrained(args.base_model)
    if base_tok.pad_token is None:
        base_tok.pad_token = base_tok.eos_token

    results = []
    for add_k in args.add_tokens_sweep:
        tok = copy.deepcopy(base_tok)
        if add_k > 0:
            tok, n_added = expand_tokenizer(tok, base_texts[:5000], add_k)
        else:
            n_added = 0
        fert = tokenizer_fertility(tok, base_texts[:500])
        print(f"\n=== fertility level: +{n_added} tokens -> fertility {fert:.3f} ===")

        model = train_condition(
            args.base_model, tok, base_texts, canaries, args.epochs, args.lr,
            os.path.join(args.out, f"ckpt_add{add_k}"))
        for r in measure(model, tok, canaries):
            r.update(add_tokens=add_k, n_added=n_added, fertility=round(fert, 3))
            results.append(r)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(os.path.join(args.out, "surface.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader(); w.writerows(results)

    # the 2-D surface: rows = fertility level, cols = reps, cell = extraction %
    from collections import defaultdict
    grid = defaultdict(list)
    for r in results:
        grid[(r["fertility"], r["reps"])].append(r["exact"])
    ferts = sorted({r["fertility"] for r in results}, reverse=True)   # high->low fertility
    reps = sorted({r["reps"] for r in results})
    print("\nEXTRACTION SURFACE  (rows: fertility high->low, cols: reps)")
    print("fert\\reps," + ",".join(map(str, reps)))
    for fv in ferts:
        cells = [f"{sum(grid[(fv, rp)]) / len(grid[(fv, rp)]):.2f}" if grid[(fv, rp)] else "-"
                 for rp in reps]
        print(f"{fv:.3f}," + ",".join(cells))
    print("\nRead: extraction should rise left->right (duplication) AND "
          "top->bottom (lower fertility). A flatter top row = high-fertility "
          "languages resist duplication-driven memorization.")


if __name__ == "__main__":
    main()
