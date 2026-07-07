"""
Shared utilities: deterministic seeding, model loading, and *correct*
conditional perplexity over a suffix.

Design notes (these matter for the paper's defensibility):
  * Perplexity is only comparable across two models when they share a
    tokenizer. We therefore expose `same_tokenizer_guard` and refuse to
    compute reference-ratio MIA across mismatched vocabularies. This is the
    single biggest bug in the v1 pipeline (InkubaLM vs Pythia have different
    tokenizers, so `ppl_ref/ppl_tgt` was uninterpretable).
  * Suffix NLL is computed *conditionally* on the prefix (prefix labels are
    masked to -100), so the score reflects memorization of the continuation,
    not fluency on the shared prompt.
"""
from __future__ import annotations
import os
import random
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(name: str, dtype: str = "bfloat16", quantize: bool = False,
               revision: str | None = None):
    """Load a causal LM + tokenizer.

    Prefer full/half precision. Quantization is a *confound* for a
    memorization study (it is itself a defense studied in the paper), so it is
    off by default and, when used, must be reported as a separate condition.

    `revision` selects a specific checkpoint (e.g. Pythia's ``step103000`` or an
    OLMo intermediate). Used by run_checkpoint_dynamics.py to trace when
    per-language memorization emerges during training.
    """
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True, revision=revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # required for correct batched generation

    kwargs = dict(trust_remote_code=True, device_map="auto", revision=revision)
    if quantize:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["torch_dtype"] = getattr(torch, dtype)

    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    model.eval()
    return model, tok


def same_tokenizer_guard(tok_a, tok_b) -> bool:
    """True iff two tokenizers produce identical ids -> ppl ratio is valid."""
    probe = "The quick brown fox jumps over the lazy dog 12345 äöå."
    return tok_a(probe)["input_ids"] == tok_b(probe)["input_ids"]


@torch.no_grad()
def suffix_nll(model, tok, prefix_text: str, suffix_text: str) -> float:
    """Mean negative log-likelihood of `suffix_text` given `prefix_text`.

    Prefix tokens are masked out of the loss (labels = -100), so this is the
    conditional NLL of the continuation only. exp(nll) == conditional PPL.
    """
    prefix_ids = tok(prefix_text, return_tensors="pt").input_ids
    suffix_ids = tok(suffix_text, return_tensors="pt", add_special_tokens=False).input_ids
    input_ids = torch.cat([prefix_ids, suffix_ids], dim=1).to(model.device)

    labels = input_ids.clone()
    labels[:, : prefix_ids.shape[1]] = -100  # do not score the shared prompt
    out = model(input_ids=input_ids, labels=labels)
    return float(out.loss)


@torch.no_grad()
def token_logprobs(model, tok, text: str):
    """Return per-token log p(x_t | x_<t) and the full-vocab log-softmax rows.

    Used by Min-K% / Min-K%++ membership inference. Returns
    (target_logprobs [T-1], mu [T-1], sigma [T-1]) where mu/sigma are the
    mean/std of the vocab log-prob distribution at each position (Min-K%++).
    """
    ids = tok(text, return_tensors="pt", truncation=True, max_length=1024).input_ids.to(model.device)
    logits = model(ids).logits[0, :-1]            # [T-1, V]
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    targets = ids[0, 1:]                            # [T-1]
    tgt_lp = logprobs[torch.arange(targets.shape[0]), targets]
    mu = logprobs.mean(dim=-1)
    sigma = logprobs.std(dim=-1)
    return tgt_lp.cpu(), mu.cpu(), sigma.cpu()
