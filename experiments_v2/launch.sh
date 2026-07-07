#!/usr/bin/env bash
# Example launcher for a multi-GPU box. Pins one language per GPU.
# For SLURM, wrap each line in `srun --gres=gpu:1` inside an sbatch array.
set -euo pipefail

OUT=results
mkdir -p $OUT

# ---- 1. Natural experiment: specialist vs base (same tokenizer) ----
# Poro-2 Finnish is the clean case: base Llama-3.1 is the reference.
for lang in finnish swahili zulu polish english; do
  CUDA_VISIBLE_DEVICES=0 python run_extraction.py \
    --target LumiOpen/Llama-Poro-2-8B-base \
    --reference meta-llama/Llama-3.1-8B \
    --lang $lang --n 2000 --seeds 0 1 2 --decoding greedy --out $OUT &
done
wait

# also the base model itself, to show extraction is INTRODUCED by specialization
for lang in finnish swahili; do
  CUDA_VISIBLE_DEVICES=0 python run_extraction.py \
    --target meta-llama/Llama-3.1-8B --lang $lang --n 2000 --seeds 0 1 2 --out $OUT &
done
wait

# ---- 2. Reference-free MIA suite (works for InkubaLM too) ----
for m in LumiOpen/Llama-Poro-2-8B-base meta-llama/Llama-3.1-8B lelapa/InkubaLM-0.4B; do
  for lang in finnish swahili zulu english; do
    python run_mia.py --model $m --lang $lang --n 2000 --out $OUT
  done
done

# ---- 3. Scaling curve (Pythia suite) ----
for size in pythia-410m pythia-1.4b pythia-2.8b pythia-6.9b; do
  for lang in english finnish swahili; do
    python run_extraction.py --target EleutherAI/$size --lang $lang \
      --n 2000 --seeds 0 1 2 --out $OUT
  done
done

# ---- 4. CAUSAL canary experiment (trains a 410M model) ----
python run_canary.py --base_model EleutherAI/pythia-410m \
  --langs english finnish swahili zulu \
  --reps 1 2 4 8 16 32 64 128 --n_per_cell 20 --out $OUT/canary

# ---- 5. NOVEL: fertility-normalized extraction + n-gram novelty floor ----
# The paper's headline decomposition (tokenization vs residual-language effect,
# and neural extraction ABOVE the n-gram floor). Char-matched spans.
for lang in english finnish swahili zulu polish; do
  python run_novel_memorization.py --model LumiOpen/Llama-Poro-2-8B-base \
    --lang $lang --n 2000 --prefix_chars 600 --suffix_chars 200 --out $OUT
done

# ---- 6. CAPSTONE: fertility intervention via vocabulary expansion ----
# Trains TWO 410M models per language (orig vs expanded tokenizer). Expensive,
# so run on the 1-2 languages that anchor the claim (a low-resource one where
# vocab expansion matters most). Content-matched by default.
for lang_cfg in "swahili swh_Latn" "zulu zul_Latn"; do
  set -- $lang_cfg
  python run_vocab_intervention.py --base_model EleutherAI/pythia-410m \
    --corpus_hf HuggingFaceFW/fineweb-2 --corpus_config $2 --lang $1 \
    --add_tokens 5000 --reps 1 2 4 8 16 32 64 --out $OUT/vocab_$1
done

# ---- 7. GRAND-UNIFIED surface: duplication x fertility (headline figure) ----
# Trains one small model per fertility level (default 4). Run on the anchor
# low-resource language. Subsumes steps 4 (canary) and 6 (vocab A/B).
python run_fertility_sweep.py --base_model EleutherAI/pythia-410m \
  --corpus_hf HuggingFaceFW/fineweb-2 --corpus_config swh_Latn --lang swahili \
  --add_tokens_sweep 0 1000 5000 20000 --reps 1 2 4 8 16 32 64 128 \
  --out $OUT/surface_swahili

# ---- 8. CROSS-LINGUAL LEAKAGE: secret injected in lang A, probed in lang B ----
# Trains one small model; privacy result. Corpus lang is just the carrier text.
python run_crosslingual_leakage.py --base_model EleutherAI/pythia-410m \
  --corpus_hf HuggingFaceFW/fineweb-2 --corpus_config swh_Latn \
  --langs english finnish swahili zulu --reps 4 16 64 --out $OUT/leak

# ---- 9. MEMORIZATION ONSET across checkpoints (open-data Pythia) ----
# Inference-only; reuses Pythia's 154 released checkpoints. One --lang per run;
# overlay the CSVs to compare onset across languages.
for lang in english finnish swahili; do
  python run_checkpoint_dynamics.py --model EleutherAI/pythia-1.4b --lang $lang \
    --n 1000 --revisions step1000 step4000 step16000 step36000 step64000 step100000 step143000 \
    --out $OUT/dynamics
done

# ---- aggregate ----
python aggregate.py --glob "$OUT/extraction_*.csv" --out tables
python aggregate.py --mia $OUT/mia_summary.csv
echo "novel decomposition    -> $OUT/novel_summary.csv"
echo "fertility intervention -> $OUT/vocab_*/vocab_intervention.csv"
echo "unified surface        -> $OUT/surface_swahili/surface.csv"
echo "cross-lingual leakage  -> $OUT/leak/leakage.csv"
echo "memorization onset     -> $OUT/dynamics/dynamics_*.csv"
