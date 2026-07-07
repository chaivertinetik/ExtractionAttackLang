"""
CAPSTONE causal experiment: intervene on tokenizer fertility, measure the
change in memorization. This is the manipulation that turns "memorization is
tokenization, not language" from a correlation into a cause.

Setup (reuses the canary machinery in run_canary.py):
  * Condition A (baseline): continue-pretrain the base model on a low-resource
    corpus + injected canaries, using the ORIGINAL tokenizer (high fertility).
  * Condition B (expanded): learn K new language-specific subword tokens from
    the SAME corpus, add them to the tokenizer, resize embeddings, then
    continue-pretrain on the SAME corpus + SAME canaries (lower fertility).
  * Inject identical canary *content* into both, so only fertility differs.

Prediction if the thesis holds:
    lower fertility (Condition B) -> HIGHER verbatim canary extraction at
    matched content, especially at low repetition counts. i.e. vocabulary
    expansion, a routine efficiency trick, silently increases memorization /
    privacy risk.

Report fertility(A) vs fertility(B) as the manipulation check, then extraction
rate vs (reps) for both conditions.

Compute: two continued-pretrains of a small base model (410M/1B). One node.

Usage:
  python run_vocab_intervention.py --base_model EleutherAI/pythia-410m \
      --corpus_hf HuggingFaceFW/fineweb-2 --corpus_config swh_Latn \
      --lang swahili --add_tokens 5000 --reps 1 2 4 8 16 32 64 --out results/vocab
"""
from __future__ import annotations
import argparse, csv, json, os, copy
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments, DataCollatorForLanguageModeling)

from common import set_seed
from run_canary import make_canaries, MixedCorpus, measure
from metrics import tokenizer_fertility


def expand_tokenizer(tok, corpus_texts, add_k):
    """Learn `add_k` new subword tokens from the corpus and splice them into
    `tok`. Returns (new_tokenizer, n_added). Standard low-resource LAPT move."""
    learned = tok.train_new_from_iterator(iter(corpus_texts), vocab_size=len(tok) + add_k * 3)
    existing = set(tok.get_vocab().keys())
    candidates = [t for t in learned.get_vocab() if t not in existing]
    # prefer the most "language-specific" (longest) new pieces
    candidates.sort(key=len, reverse=True)
    n_added = tok.add_tokens(candidates[:add_k])
    return tok, n_added


def train_condition(base_model, tokenizer, base_texts, canaries, epochs, lr, out):
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16)
    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))   # mean-init new rows
    dataset = MixedCorpus(tokenizer, base_texts, canaries)
    Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=out, per_device_train_batch_size=8,
            gradient_accumulation_steps=4, num_train_epochs=epochs,
            learning_rate=lr, bf16=True, logging_steps=50,
            save_strategy="no", report_to=[],
        ),
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    ).train()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="EleutherAI/pythia-410m")
    ap.add_argument("--corpus_hf", default="HuggingFaceFW/fineweb-2")
    ap.add_argument("--corpus_config", default="swh_Latn")
    ap.add_argument("--corpus_docs", type=int, default=20000)
    ap.add_argument("--lang", default="swahili")
    ap.add_argument("--add_tokens", type=int, default=5000)
    ap.add_argument("--reps", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--n_per_cell", type=int, default=25)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--token_matched", action="store_true",
                    help="scale Condition B epochs by fertility ratio so both "
                         "conditions see the same number of TRAINING TOKENS "
                         "(controls for compute; default holds CONTENT fixed)")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/vocab")
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

    canaries = make_canaries([args.lang], args.reps, args.n_per_cell, args.seed)
    json.dump(canaries, open(os.path.join(args.out, "canaries.json"), "w"))

    tok_a = AutoTokenizer.from_pretrained(args.base_model)
    if tok_a.pad_token is None:
        tok_a.pad_token = tok_a.eos_token
    tok_b = copy.deepcopy(tok_a)
    tok_b, n_added = expand_tokenizer(tok_b, base_texts[:5000], args.add_tokens)

    fert_a = tokenizer_fertility(tok_a, base_texts[:500])
    fert_b = tokenizer_fertility(tok_b, base_texts[:500])
    print(f"[manipulation check] fertility A(orig)={fert_a:.3f}  "
          f"B(+{n_added} tok)={fert_b:.3f}  ratio={fert_a / fert_b:.3f}")

    epochs_b = args.epochs * (fert_a / fert_b) if args.token_matched else args.epochs

    results = []
    for cond, tok, ep in [("A_orig", tok_a, args.epochs), ("B_expanded", tok_b, epochs_b)]:
        print(f"\n=== Condition {cond} (fertility "
              f"{fert_a if cond.startswith('A') else fert_b:.3f}, epochs {ep:.2f}) ===")
        model = train_condition(args.base_model, tok, base_texts, canaries, ep, args.lr,
                                os.path.join(args.out, f"ckpt_{cond}"))
        for r in measure(model, tok, canaries):
            r.update(condition=cond,
                     fertility=round(fert_a if cond.startswith("A") else fert_b, 3))
            results.append(r)
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    with open(os.path.join(args.out, "vocab_intervention.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader(); w.writerows(results)

    # the money table: extraction rate vs reps, per condition
    from collections import defaultdict
    agg = defaultdict(list)
    for r in results:
        agg[(r["condition"], r["reps"])].append(r["exact"])
    print("\ncondition,reps,exact_rate")
    for (cond, reps), v in sorted(agg.items()):
        print(f"{cond},{reps},{sum(v) / len(v):.3f}")
    print("\nIf B_expanded > A_orig at matched reps -> lowering fertility via "
          "vocabulary expansion INCREASES memorization (thesis confirmed).")


if __name__ == "__main__":
    main()
