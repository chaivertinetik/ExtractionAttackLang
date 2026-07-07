# Multilingual Memorization v2 — experiment package (ACL-track)

Corrected + extended pipeline for the multilingual training-data-memorization
paper. Fixes the methodological issues in the v1 scripts and adds the
experiments a top-tier venue will expect.

## What changed vs v1 (`../new_main.py`) and why it matters

| v1 problem | Consequence | Fix (here) |
|---|---|---|
| `ppl_ratio = ppl_ref/ppl_tgt` across **different tokenizers** (InkubaLM vs Pythia) | Ratio is uninterpretable; the whole InkubaLM arm is null | `same_tokenizer_guard` blocks it; use reference-free **Min-K%++** MIA instead (`run_mia.py`) |
| Perplexity over prompt+suffix | Score reflects prompt fluency, not memorized continuation | **Conditional suffix NLL** (prefix masked to `-100`) in `common.suffix_nll` |
| `is_mosaic`, `is_exact` from magic char-level thresholds (2.0 / 0.45 / 0.95) | Reviewer rejects unvalidated metric | Token-level **exact match** + prefix-match-length distribution + human-audited edit-sim |
| No non-member control set | Can't tell "memorized" from "fluent" | Temporal/held-out **non-member** set per language; report **AUROC + TPR@1%FPR** |
| Random char offsets across concatenated docs | Prompts splice unrelated documents | One `(prefix, suffix)` pair **per document**, token-aligned (`data.make_pairs`) |
| 4-bit quantized target | Quantization is itself a *defense* you study → confound | Full/bf16 by default; quantization is a separate reported condition |
| Single run, no error bars | Not publishable | 3 seeds + **bootstrap 95% CIs** (`aggregate.py`) |
| Greedy only | Only a lower bound | greedy **and** top-k sampling attack |

## The story the data actually supports right now
Across your v1 CSVs the only clean signal is the **Poro-2 natural experiment**:
the Finnish continued-pretrain assigns much lower perplexity + produces
near-duplicate continuations **specifically on Finnish** vs the base Llama-3.1,
and is flat on control languages (pl/zu/wo/ne). Verbatim `is_exact` is ~0
everywhere, so a paper cannot rest on verbatim extraction alone — it must
either (a) make MIA the headline (this package), and/or (b) run the causal
canary experiment to *manufacture* controlled memorization and measure how
language modulates it.

## Experiments (in priority order for the paper)
1. **Natural experiment — specialist vs base** (`run_extraction.py`, arm 1 in
   `launch.sh`). Poro-2 vs Llama-3.1 on Finnish + controls. Clean because same
   arch/tokenizer, known added data.
2. **Reference-free MIA suite** (`run_mia.py`). AUROC + TPR@1%FPR per language;
   works for InkubaLM. Correlate AUC with **tokenizer fertility** (mechanism).
3. **Scaling curve** (Pythia/OLMo suites). The log-linear memorization-vs-size
   curve, per language — the modern update of the v1 Figure 6/7.
4. **Causal canary injection** (`run_canary.py`). Inject canaries at controlled
   repetition counts across languages, continue-pretrain a 410M model, measure
   extraction rate vs (language, reps). Turns the correlational "morphology →
   less memorization" claim into a causal one with everything else held fixed.
5. **NOVEL — "memorization is tokenization, not language"**
   (`run_novel_memorization.py`) — **the paper's original contribution.** Two
   controls no prior multilingual-memorization work applies:
   * *Fertility-normalized extraction*: measure at matched CHARACTER spans, not
     fixed token counts, and log tokenizer fertility per sample. Lets you
     decompose the cross-language gap into a tokenization component and a
     residual language component. Likely re-explains the 2024 U-shaped curve.
   * *N-gram novelty floor*: a cheap word n-gram on the same corpus greedily
     extracts the suffix; true memorization = neural − n-gram floor. Essential
     for low-resource langs, whose tiny repetitive corpora make a dumb n-gram
     "extract" text the model never memorized. Validated: on a repetitive
     corpus the floor is high (so raw neural numbers are misleading); on
     high-entropy text the floor is ~0.

## Setup
```bash
pip install torch transformers datasets accelerate pyyaml numpy tqdm
huggingface-cli login   # Llama-3.1 / Poro-2 are gated
```
Edit `configs.yaml` to confirm member/non-member sources (verify Poro-2's
Finnish corpus + InkubaLM splits before publishing — provenance is load-bearing).

## Run
```bash
bash launch.sh                 # or submit as a SLURM array, one lang per GPU
python aggregate.py --glob 'results/extraction_*.csv' --out tables
```

## Compute notes
Inference arms: 1×A100-80GB (bf16) per 8B model; 410M/1.4B fit anywhere.
Canary arm trains a 410M model — one A100, a few hours for the default budget.
The stack is PyTorch/CUDA. If your cluster is genuinely **TPU**, either use the
GPU nodes for this, or port `run_canary.py` to `torch_xla` / a JAX trainer
(the inference scripts would need JAX reimplementation — flag if you need it).

## Files
- `common.py` — loading, seeding, conditional suffix NLL, per-token logprobs
- `metrics.py` — exact match, edit sim, Min-K%/Min-K%++, zlib, fertility, AUC, TPR@FPR
- `data.py` — doc-boundary member/non-member pair construction
- `run_extraction.py` — prefix/suffix extraction attack
- `run_mia.py` — membership inference suite + summary
- `run_canary.py` — causal canary-injection continued-pretraining
- `aggregate.py` — bootstrap-CI tables + LaTeX
- `configs.yaml` — models + per-language data sources
- `launch.sh` — end-to-end example
