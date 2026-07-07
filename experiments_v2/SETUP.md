# Setup & Run Guide (hand this to whoever runs the experiments)

This is the **v2** pipeline (folder `experiments_v2/`). It replaces the old
`data_batch*.py` / `hatch run` flow.

### What's different from the old repo
- **No Google Drive download.** Data is streamed directly from HuggingFace at
  run time (see `configs.yaml`). There is no `Data_Extraction_data/` folder.
- **You must log in to HuggingFace and get access to gated models**
  (Llama-3.1-8B, Llama-Poro-2-8B). The old Pythia/GPT-Neo runs were ungated.
- **GPU, not TPU.** bf16. An 8B model needs ~1×A100-80GB; 410M/1.4B run on
  anything. Two scripts *train* a model (see step 6).

---

## 1. Clone and enter the folder
```bash
git clone https://github.com/pchaitanya21/ExtractionAttackLang.git
cd ExtractionAttackLang/experiments_v2
```

## 2. Environment
```bash
python -m venv .venv && source .venv/bin/activate      # or conda
pip install -r requirements.txt
```

## 3. HuggingFace auth + gated model access (one time)
```bash
huggingface-cli login          # paste a token from https://huggingface.co/settings/tokens
```
Then click "Agree/Request access" on each gated model page (approval is
usually instant):
- https://huggingface.co/meta-llama/Llama-3.1-8B
- https://huggingface.co/LumiOpen/Llama-Poro-2-8B-base

(Pythia, OLMo, InkubaLM, FineWeb-2, C4 are open — no request needed.)

## 4. Smoke test (do this FIRST — ~2 min, no GPU/gated model needed)
Confirms the env works before you spend real compute:
```bash
python run_mia.py --model EleutherAI/pythia-160m --lang english --n 20 --out results
```
You should see a line like `[done] {'model': 'pythia-160m_english', ... 'auc_min_k_pp': 0.5x ...}`
and a `results/mia_pythia-160m_english.csv`. If that works, the pipeline is good.

## 5. Verify the data sources (IMPORTANT before real runs)
Open `configs.yaml` and confirm the `member` / `nonmember` datasets for each
language are the corpora these models were actually trained on. The defaults
are reasonable placeholders (FineWeb-2 shards, Inkuba-Mono, C4) but the paper's
claims depend on getting membership right. Ask Chai if unsure.

## 6. Run the experiments
Everything at once (edit GPU ids / languages inside first):
```bash
chmod +x launch.sh
./launch.sh
```
Or one experiment at a time — **inference-only** (fast, any GPU):
```bash
# corrected extraction attack (specialist vs its base)
python run_extraction.py --target LumiOpen/Llama-Poro-2-8B-base \
  --reference meta-llama/Llama-3.1-8B --lang finnish --n 2000 --seeds 0 1 2 --out results

# membership inference suite (works for InkubaLM too)
python run_mia.py --model lelapa/InkubaLM-0.4B --lang swahili --n 2000 --out results

# NOVEL: fertility-normalized extraction + n-gram floor
python run_novel_memorization.py --model LumiOpen/Llama-Poro-2-8B-base \
  --lang finnish --n 2000 --prefix_chars 600 --suffix_chars 200 --out results
```
**Training runs** (need a real GPU for a few hours each):
```bash
# causal canary injection (trains one 410M model)
python run_canary.py --base_model EleutherAI/pythia-410m \
  --langs english finnish swahili zulu --reps 1 2 4 8 16 32 64 128 --out results/canary

# CAPSTONE: fertility intervention (trains TWO 410M models)
python run_vocab_intervention.py --base_model EleutherAI/pythia-410m \
  --corpus_hf HuggingFaceFW/fineweb-2 --corpus_config swh_Latn --lang swahili \
  --add_tokens 5000 --reps 1 2 4 8 16 32 64 --out results/vocab_swahili
```

## 7. Collect results into tables
```bash
python aggregate.py --glob 'results/extraction_*.csv' --out tables   # -> tables/extraction.tex
python aggregate.py --mia results/mia_summary.csv
cat results/novel_summary.csv          # tokenization-vs-language decomposition
cat results/vocab_*/vocab_intervention.csv
```
All raw per-sample CSVs land in `results/`; summary tables in `tables/`.

## 8. If you hit GPU OOM (the old "batch size 1000 → 50" note)
- Lower `--batch_size` on `run_extraction.py` (default 16 → try 8 or 4).
- Lower `--n` for a quick pass (e.g. `--n 500`).
- For the 8B models, use a single A100-80GB; do **not** use `--quantize` for
  the reported numbers (quantization is a *defense* we study separately — it
  confounds the memorization measurement). `--quantize` exists only for a
  separate ablation.
- The training scripts: reduce `per_device_train_batch_size` inside the script,
  or `--corpus_docs` / `--n_per_cell` for a smaller run.

## Which script does what
| Script | Type | Trains? |
|---|---|---|
| `run_extraction.py` | prefix/suffix extraction attack | no |
| `run_mia.py` | membership inference (Min-K%++ etc.) | no |
| `run_novel_memorization.py` | fertility-normalized + n-gram floor | no |
| `run_canary.py` | causal canary injection | **yes** (410M) |
| `run_vocab_intervention.py` | fertility intervention capstone | **yes** (2×410M) |
| `aggregate.py` | build CI tables + LaTeX | no |
