"""
CAUSAL canary-injection experiment -- the centerpiece for an ACL submission.

Observational cross-model comparisons (v1) can only *correlate* language with
memorization; a reviewer will object that language, corpus, dedup, and
frequency are all confounded. The fix is a controlled experiment:

  1. Take a base model (e.g. EleutherAI/pythia-410m or a 1B) and a clean
     continued-pretraining corpus.
  2. Inject synthetic "canary" sequences (random but grammatical-looking
     strings) in several languages, each at a controlled repetition count
     r in {1, 2, 4, 8, 16, 32, 64, 128}.
  3. Continue-pretrain for a fixed budget.
  4. Measure extraction / MIA of each canary as a function of (language, r).

This lets you make *causal* claims:
  * "Verbatim memorization grows log-linearly with duplication count, but the
    slope is significantly lower for morphologically rich languages
    (Finnish/Zulu) than for English/Swahili at matched r and matched token
    budget."  <- this is the paper's money result and it is confound-free
    because r, corpus, and training are held identical across languages.
  * Correlate the per-language slope with tokenizer fertility.

Canaries are held out of any released data; only aggregate rates are reported.

Run on a single high-mem GPU (410M) or multi-GPU (1B+). This is the one script
that trains; everything else is inference-only.
"""
from __future__ import annotations
import argparse, json, os, random
import torch
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          Trainer, TrainingArguments, DataCollatorForLanguageModeling)

from common import set_seed
from metrics import token_exact_match, token_prefix_match_len


# --- 1. Build canaries -------------------------------------------------------
# Templated, language-flavored nonsense so canaries are in-distribution for the
# tokenizer but guaranteed absent from the base corpus. Fill lexicons per lang.
LEXICONS = {
    "english": "the quiet harbor sold amber lanterns before winter arrived again".split(),
    "finnish": "hiljainen satama myi meripihkalyhtyjä ennen talven paluuta jälleen".split(),
    "swahili": "bandari tulivu iliuza taa za kahawia kabla ya majira ya baridi".split(),
    "zulu":    "itheku elithulile lidayise izibani zombala phambi kobusika bafika".split(),
}


def make_canaries(langs, reps, n_per_cell, seed=0):
    rng = random.Random(seed)
    canaries = []
    for lang in langs:
        vocab = LEXICONS[lang]
        for r in reps:
            for j in range(n_per_cell):
                seq = " ".join(rng.choices(vocab, k=30))
                canaries.append(dict(lang=lang, reps=r, idx=j, text=seq))
    return canaries


# --- 2. Dataset that interleaves corpus + repeated canaries ------------------
class MixedCorpus(Dataset):
    def __init__(self, tokenizer, base_texts, canaries, block=256):
        self.tok = tokenizer
        docs = list(base_texts)
        for c in canaries:                      # inject each canary `reps` times
            docs += [c["text"]] * c["reps"]
        random.shuffle(docs)
        ids = []
        for d in docs:
            ids += tokenizer(d, add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
        self.blocks = [ids[i:i + block] for i in range(0, len(ids) - block, block)]

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, i):
        return {"input_ids": torch.tensor(self.blocks[i])}


# --- 3. Measure extraction of canaries after training -----------------------
@torch.no_grad()
def measure(model, tok, canaries, prefix_words=10, gen_tokens=40):
    model.eval()
    out = []
    for c in canaries:
        words = c["text"].split()
        prefix = " ".join(words[:prefix_words])
        true_ids = tok(" ".join(words[prefix_words:]), add_special_tokens=False).input_ids
        enc = tok(prefix, return_tensors="pt").to(model.device)
        gen = model.generate(**enc, max_new_tokens=gen_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        g_ids = gen[0, enc.input_ids.shape[1]:].tolist()
        out.append(dict(lang=c["lang"], reps=c["reps"],
                        exact=token_exact_match(g_ids, true_ids, len(true_ids)),
                        pml=token_prefix_match_len(g_ids, true_ids, len(true_ids))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="EleutherAI/pythia-410m")
    ap.add_argument("--corpus_hf", default="allenai/c4")
    ap.add_argument("--corpus_config", default="en")
    ap.add_argument("--corpus_docs", type=int, default=20000)
    ap.add_argument("--langs", nargs="+", default=["english", "finnish", "swahili", "zulu"])
    ap.add_argument("--reps", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128])
    ap.add_argument("--n_per_cell", type=int, default=20)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/canary")
    args = ap.parse_args()
    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16)

    from datasets import load_dataset
    stream = load_dataset(args.corpus_hf, args.corpus_config, split="train", streaming=True)
    base_texts = []
    for row in stream:
        base_texts.append(row["text"])
        if len(base_texts) >= args.corpus_docs:
            break

    canaries = make_canaries(args.langs, args.reps, args.n_per_cell, args.seed)
    json.dump(canaries, open(os.path.join(args.out, "canaries.json"), "w"))
    dataset = MixedCorpus(tok, base_texts, canaries)

    Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=os.path.join(args.out, "ckpt"),
            per_device_train_batch_size=8, gradient_accumulation_steps=4,
            num_train_epochs=args.epochs, learning_rate=args.lr,
            bf16=True, logging_steps=50, save_strategy="no", report_to=[],
        ),
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
    ).train()

    results = measure(model, tok, canaries)
    import csv
    with open(os.path.join(args.out, "canary_extraction.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader(); w.writerows(results)

    # per (lang, reps) extraction rate -> the log-linear curves
    from collections import defaultdict
    agg = defaultdict(list)
    for r in results:
        agg[(r["lang"], r["reps"])].append(r["exact"])
    print("lang,reps,exact_rate")
    for (lang, reps), v in sorted(agg.items()):
        print(f"{lang},{reps},{sum(v)/len(v):.3f}")


if __name__ == "__main__":
    main()
