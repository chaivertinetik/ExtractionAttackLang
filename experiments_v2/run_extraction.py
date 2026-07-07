"""
Prefix/suffix EXTRACTION attack (corrected).

Headline metric  : token-level exact match of the 50-token suffix (greedy).
Companion metrics : prefix-match length distribution, edit similarity,
                    conditional suffix-PPL under target and (if same tokenizer)
                    reference model.

Natural-experiment design
-------------------------
For the Poro-2 (Finnish continued-pretrain of Llama-3.1-8B) case the *base*
model is the ideal reference: identical architecture + tokenizer, minus the
Finnish specialization data. Extraction that appears in the specialist but not
the base is memorization *introduced by specialization* -- a clean causal-ish
claim. Same tokenizer => the reference PPL ratio is valid here (guarded).

Usage:
  python run_extraction.py --target LumiOpen/Llama-Poro-2-8B-base \
      --reference meta-llama/Llama-3.1-8B --lang finnish \
      --n 2000 --seeds 0 1 2 --decoding greedy --out results/

Run one language per process; parallelize across GPUs with the launcher.
"""
from __future__ import annotations
import argparse, csv, os
import torch
from tqdm import tqdm

from common import load_model, set_seed, same_tokenizer_guard, suffix_nll
from data import make_pairs
from metrics import token_exact_match, token_prefix_match_len, edit_similarity, zlib_entropy
import yaml


@torch.no_grad()
def generate_suffix(model, tok, prefix_texts, suffix_len, decoding):
    enc = tok(prefix_texts, return_tensors="pt", padding=True).to(model.device)
    gen_kwargs = dict(max_new_tokens=suffix_len, pad_token_id=tok.eos_token_id)
    if decoding == "greedy":
        gen_kwargs.update(do_sample=False)
    elif decoding == "topk":            # Carlini'21-style sampling attack
        gen_kwargs.update(do_sample=True, top_k=40, temperature=1.0)
    out = model.generate(**enc, **gen_kwargs)
    gen_only = out[:, enc.input_ids.shape[1]:]
    return gen_only


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--reference", default=None)
    ap.add_argument("--lang", required=True)
    ap.add_argument("--config", default="configs.yaml")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--prefix_len", type=int, default=150)
    ap.add_argument("--suffix_len", type=int, default=50)
    ap.add_argument("--decoding", choices=["greedy", "topk"], default="greedy")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--quantize", action="store_true")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    member_src = cfg["languages"][args.lang]["member"]

    tgt, tgt_tok = load_model(args.target, quantize=args.quantize)
    ref = ref_tok = None
    ratio_valid = False
    if args.reference:
        ref, ref_tok = load_model(args.reference, quantize=args.quantize)
        ratio_valid = same_tokenizer_guard(tgt_tok, ref_tok)
        if not ratio_valid:
            print("[warn] target/reference tokenizers differ -> PPL ratio "
                  "column will be NaN (uninterpretable). Use MIA suite instead.")

    os.makedirs(args.out, exist_ok=True)
    tag = f"{args.target.split('/')[-1]}_{args.lang}_{args.decoding}"
    rows = []

    for seed in args.seeds:
        set_seed(seed)
        pairs = list(make_pairs(tgt_tok, member_src, args.n, is_member=1,
                                prefix_len=args.prefix_len, suffix_len=args.suffix_len))
        for i in tqdm(range(0, len(pairs), args.batch_size), desc=f"{args.lang} seed{seed}"):
            batch = pairs[i:i + args.batch_size]
            gen = generate_suffix(tgt, tgt_tok, [p.prefix_text for p in batch],
                                  args.suffix_len, args.decoding)
            for p, g_ids in zip(batch, gen):
                g_ids = g_ids.tolist()
                gen_text = tgt_tok.decode(g_ids, skip_special_tokens=True)
                exact = token_exact_match(g_ids, p.suffix_ids, args.suffix_len)
                pml = token_prefix_match_len(g_ids, p.suffix_ids, args.suffix_len)
                ppl_tgt = suffix_nll(tgt, tgt_tok, p.prefix_text, p.suffix_text)
                ppl_ref = suffix_nll(ref, ref_tok, p.prefix_text, p.suffix_text) if ratio_valid else float("nan")
                rows.append(dict(
                    seed=seed, doc_id=p.doc_id,
                    is_exact=exact, prefix_match_len=pml,
                    edit_sim=round(edit_similarity(p.suffix_text, gen_text), 4),
                    suffix_nll_tgt=round(ppl_tgt, 4),
                    suffix_nll_ref=round(ppl_ref, 4) if ratio_valid else "",
                    nll_gap=round(ppl_ref - ppl_tgt, 4) if ratio_valid else "",
                    zlib=zlib_entropy(gen_text),
                ))

    out_csv = os.path.join(args.out, f"extraction_{tag}.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    n_exact = sum(r["is_exact"] for r in rows)
    print(f"[done] {out_csv}: {n_exact}/{len(rows)} exact "
          f"({100 * n_exact / len(rows):.3f}%)")


if __name__ == "__main__":
    main()
