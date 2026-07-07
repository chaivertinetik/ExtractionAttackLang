"""
Memorization + membership-inference metrics.

Verbatim / approximate extraction:
  * token_exact_match  -> standard "extractable memorization" (Carlini'22,
    Nasr'23): the model reproduces the true continuation token-for-token.
  * edit_similarity    -> normalized edit distance (approximate memorization).
  * We deliberately drop the character-level SequenceMatcher>0.95 heuristic
    from v1; report token-level exact match as the headline number and edit
    similarity as the "approximate" companion, with a human-validated
    threshold (see aggregate.py / --human_audit).

Membership inference (no reference model needed -> works for InkubaLM too):
  * loss / conditional NLL
  * zlib-calibrated loss (Carlini'21)
  * Min-K% Prob (Shi et al. 2023)
  * Min-K%++ (Zhang et al. 2024)   <- current SOTA reference-free MIA
Reference-ratio MIA is provided in mia.py but is *only* valid when the two
models share a tokenizer (guarded).
"""
from __future__ import annotations
import zlib
import numpy as np
import torch


def token_exact_match(gen_ids, true_ids, k: int = 50) -> int:
    """1 iff the first k generated tokens equal the true suffix tokens."""
    g = list(gen_ids)[:k]
    t = list(true_ids)[:k]
    return int(len(g) == len(t) and g == t)


def token_prefix_match_len(gen_ids, true_ids, k: int = 50) -> int:
    """Length of the longest matching *prefix* of the two token sequences.

    Reporting the distribution of this (0..k) is far more informative than a
    single exact/not-exact bit and is robust to a single divergent token.
    """
    n = 0
    for a, b in zip(list(gen_ids)[:k], list(true_ids)[:k]):
        if a != b:
            break
        n += 1
    return n


def edit_similarity(a: str, b: str) -> float:
    """1 - normalized Levenshtein distance over characters (0..1)."""
    if not a and not b:
        return 1.0
    la, lb = len(a), len(b)
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, lb + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return 1.0 - dp[lb] / max(la, lb)


def zlib_entropy(text: str) -> int:
    return len(zlib.compress(text.encode("utf-8")))


def min_k_percent(tgt_lp: torch.Tensor, k: float = 0.2) -> float:
    """Min-K% Prob: mean of the k-fraction lowest token log-probs."""
    n = max(1, int(len(tgt_lp) * k))
    return float(torch.topk(tgt_lp, n, largest=False).values.mean())


def min_k_percent_pp(tgt_lp, mu, sigma, k: float = 0.2) -> float:
    """Min-K%++: standardize each token log-prob by the vocab distribution
    at that position, then average the k-fraction lowest standardized scores."""
    z = (tgt_lp - mu) / (sigma + 1e-8)
    n = max(1, int(len(z) * k))
    return float(torch.topk(z, n, largest=False).values.mean())


def tokenizer_fertility(tokenizer, texts) -> float:
    """Mean tokens-per-whitespace-word. Operationalizes 'morphological
    complexity' -> agglutinative langs (Finnish/Turkish/Zulu) fragment into
    more subwords, which we hypothesize *reduces* verbatim memorization."""
    tot_tok = tot_word = 0
    for t in texts:
        w = len(t.split())
        if w == 0:
            continue
        tot_tok += len(tokenizer(t, add_special_tokens=False).input_ids)
        tot_word += w
    return tot_tok / max(1, tot_word)


def auc(member_scores, nonmember_scores) -> float:
    """AUROC of a membership score (higher score => more likely member).

    Uses *average* ranks for ties (Mann-Whitney U). Ordinal ranks would bias
    the score whenever member/non-member values coincide (common for discrete
    MIA scores), e.g. identical distributions must give 0.5, not < 0.5.
    """
    m = np.asarray(member_scores, float)
    n = np.asarray(nonmember_scores, float)
    if len(m) == 0 or len(n) == 0:
        return float("nan")
    all_s = np.concatenate([m, n])
    order = all_s.argsort(kind="mergesort")
    sorted_s = all_s[order]
    ranks = np.empty(len(all_s), float)
    i = 0
    while i < len(sorted_s):                    # assign average rank to ties
        j = i
        while j < len(sorted_s) and sorted_s[j] == sorted_s[i]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    r_m = ranks[: len(m)].sum()
    return float((r_m - len(m) * (len(m) + 1) / 2) / (len(m) * len(n)))


def tpr_at_low_fpr(member_scores, nonmember_scores, fpr: float = 0.01) -> float:
    """TPR @ fixed low FPR -- the metric the MIA literature now reports,
    because average AUC hides whether a *few* points are confidently exposed."""
    m = np.asarray(member_scores, float)
    n = np.asarray(nonmember_scores, float)
    if len(m) == 0 or len(n) == 0:
        return float("nan")
    thresh = np.quantile(n, 1 - fpr)
    return float((m >= thresh).mean())
