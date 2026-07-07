"""
NOVEL contribution: "Memorization is tokenization, not language."

Two things no prior multilingual-memorization paper does, both here:

(A) FERTILITY-NORMALIZED extraction.
    Everyone (incl. the 2024 paper) fixes a 150-token prefix / 50-token
    suffix. But tokenizer fertility (subwords per word) varies 2-3x across
    languages, so "50 tokens" is much less *content* in Finnish than English.
    We instead cut prefix/suffix at matched CHARACTER spans (snapped to word
    boundaries) so the amount of real content is equal across languages, and
    we log the resulting token length so the tokenization effect is explicit.
    This lets the paper decompose the cross-language memorization gap into a
    tokenization component and a residual language component.

(B) N-GRAM NOVELTY FLOOR.
    For small low-resource corpora, "the model reproduced the suffix" is often
    just "there was only one plausible continuation." We build a cheap word
    n-gram model (stupid backoff) on the *same* sampled corpus and use it to
    (i) greedily extract the suffix and (ii) score the suffix log-prob. True
    memorization := neural extraction MINUS what the n-gram already predicts.
    Without this control the low-resource memorization numbers are not
    trustworthy.

Output per sample: neural_exact, ngram_exact, edit_sim, fertility,
ngram_logprob, char_len, tok_len. Summary prints, per language:
    neural exact %, n-gram exact % (floor), EXCESS (neural - ngram), fertility.

The n-gram half is pure-Python (no transformers) so it can be validated
standalone; pass --no_neural to run only the floor.

Usage:
  python run_novel_memorization.py --model LumiOpen/Llama-Poro-2-8B-base \
      --lang finnish --n 2000 --prefix_chars 600 --suffix_chars 200 --out results/
"""
from __future__ import annotations
import argparse, csv, math, os, random
from collections import defaultdict, Counter


# ----------------------------------------------------------------------------
# Character-matched (fertility-normalized) prefix/suffix construction
# ----------------------------------------------------------------------------
def char_pairs(texts, prefix_chars, suffix_chars, min_chars):
    """Yield (prefix, suffix) cut at matched character spans, snapped to word
    boundaries so we never split a word across the prefix/suffix seam."""
    for text in texts:
        text = " ".join(text.split())          # normalize whitespace
        if len(text) < min_chars:
            continue
        # snap the prefix end to the next space at/after prefix_chars
        cut = text.find(" ", prefix_chars)
        if cut == -1:
            continue
        end = text.find(" ", cut + suffix_chars)
        end = end if end != -1 else len(text)
        prefix, suffix = text[:cut], text[cut + 1:end]
        if len(suffix) < suffix_chars * 0.6:
            continue
        yield prefix, suffix


# ----------------------------------------------------------------------------
# (B) Word n-gram model with stupid backoff  -- the novelty floor
# ----------------------------------------------------------------------------
class NGram:
    def __init__(self, order=5, alpha=0.4):
        self.order = order
        self.alpha = alpha
        self.ctx = [defaultdict(Counter) for _ in range(order)]  # ctx[k]: k-word context
        self.unigram = Counter()
        self.total = 0

    def fit(self, docs):
        for words in docs:
            self.unigram.update(words)
            self.total += len(words)
            for i in range(len(words)):
                for k in range(1, self.order):
                    if i - k < 0:
                        break
                    self.ctx[k][tuple(words[i - k:i])][words[i]] += 1
        return self

    def _next_dist(self, history):
        """Highest-order non-empty context available for `history`."""
        for k in range(min(self.order - 1, len(history)), 0, -1):
            c = self.ctx[k].get(tuple(history[-k:]))
            if c:
                return c
        return self.unigram

    def generate(self, prefix_words, n_words):
        hist = list(prefix_words)
        out = []
        for _ in range(n_words):
            dist = self._next_dist(hist)
            nxt = dist.most_common(1)[0][0]
            out.append(nxt)
            hist.append(nxt)
        return out

    def logprob(self, prefix_words, suffix_words):
        """Stupid-backoff log-prob of the suffix given the prefix (base e)."""
        hist = list(prefix_words)
        lp = 0.0
        for w in suffix_words:
            prob, backoff = None, 1.0
            for k in range(min(self.order - 1, len(hist)), 0, -1):
                c = self.ctx[k].get(tuple(hist[-k:]))
                if c and c.total() > 0:
                    if c[w] > 0:
                        prob = backoff * c[w] / c.total()
                        break
                    backoff *= self.alpha
            if prob is None:
                prob = backoff * (self.unigram[w] + 1) / (self.total + len(self.unigram) + 1)
            lp += math.log(prob)
            hist.append(w)
        return lp


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--lang", required=True)
    ap.add_argument("--config", default="configs.yaml")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--prefix_chars", type=int, default=600)
    ap.add_argument("--suffix_chars", type=int, default=200)
    ap.add_argument("--ngram_order", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_neural", action="store_true")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    import yaml
    from data import _stream
    cfg = yaml.safe_load(open(args.config))
    src = cfg["languages"][args.lang]["member"]

    # collect a corpus (n docs) and build char-matched pairs from it
    texts = []
    for t in _stream(src):
        if len(t.split()) >= 40:
            texts.append(t)
        if len(texts) >= args.n:
            break
    pairs = list(char_pairs(texts, args.prefix_chars, args.suffix_chars,
                            args.prefix_chars + args.suffix_chars))

    # fit the n-gram floor on the SAME sampled corpus
    ngram = NGram(order=args.ngram_order).fit([t.split() for t in texts])

    # neural model (optional)
    model = tok = None
    if not args.no_neural and args.model:
        from common import load_model
        model, tok = load_model(args.model)

    from metrics import edit_similarity, tokenizer_fertility
    rows = []
    for prefix, suffix in pairs:
        pw, sw = prefix.split(), suffix.split()

        # (B) n-gram floor: greedy extraction + suffix log-prob
        ng_gen = " ".join(ngram.generate(pw, len(sw)))
        ngram_exact = int(" ".join(sw) == ng_gen)
        ng_lp = ngram.logprob(pw, sw)

        row = dict(lang=args.lang, char_len=len(suffix), word_len=len(sw),
                   ngram_exact=ngram_exact, ngram_logprob=round(ng_lp, 3))

        # (A) fertility-normalized neural extraction
        if model is not None:
            import torch
            enc = tok(prefix, return_tensors="pt").to(model.device)
            budget = int(len(suffix) / 2) + 16          # generous token budget
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=budget, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            gen_text = tok.decode(gen[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
            gen_text = " ".join(gen_text.split())[:len(suffix)]     # match char span
            row["tok_len"] = len(tok(suffix, add_special_tokens=False).input_ids)
            row["fertility"] = round(row["tok_len"] / max(1, len(sw)), 3)
            row["neural_exact"] = int(gen_text.strip() == suffix.strip())
            row["edit_sim"] = round(edit_similarity(suffix, gen_text), 4)
        rows.append(row)

    tag = f"{(args.model or 'ngramonly').split('/')[-1]}_{args.lang}"
    with open(os.path.join(args.out, f"novel_{tag}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader(); w.writerows(rows)

    # summary: the decomposition table the paper needs
    def rate(key):
        vals = [r[key] for r in rows if key in r]
        return 100 * sum(vals) / len(vals) if vals else float("nan")
    ng = rate("ngram_exact")
    nn = rate("neural_exact")
    fert = (sum(r["fertility"] for r in rows if "fertility" in r) /
            max(1, sum("fertility" in r for r in rows)))
    print(f"[{args.lang}] n={len(rows)} | fertility={fert:.2f} "
          f"| ngram_floor={ng:.2f}% | neural={nn:.2f}% "
          f"| EXCESS(true mem)={nn - ng:.2f}%")

    sfile = os.path.join(args.out, "novel_summary.csv")
    hdr = not os.path.exists(sfile)
    with open(sfile, "a", newline="") as f:
        w = csv.writer(f)
        if hdr:
            w.writerow(["model", "lang", "n", "fertility", "ngram_floor_pct",
                        "neural_pct", "excess_pct"])
        w.writerow([tag, args.lang, len(rows), round(fert, 3),
                    round(ng, 3), round(nn, 3), round(nn - ng, 3)])


if __name__ == "__main__":
    main()
