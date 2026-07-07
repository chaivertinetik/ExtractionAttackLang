"""
Data loading for member / non-member sets.

Two critical fixes over v1:
  1. Respect document boundaries. v1 concatenated all docs into one string and
     cut at random *character* offsets, splicing across documents. We tokenize
     each document and take one (prefix, suffix) pair per document.
  2. Provide a genuine NON-MEMBER control set (temporal holdout: data created
     after the model's training cutoff, or a held-out language shard). MIA and
     the "is this memorized vs just fluent" question are meaningless without it.

`member_source` / `nonmember_source` are (hf_dataset, config, split) tuples or
local jsonl paths. Adapt in configs.yaml.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator
from datasets import load_dataset


@dataclass
class Pair:
    doc_id: str
    prefix_ids: list
    suffix_ids: list
    prefix_text: str
    suffix_text: str
    is_member: int


def _stream(source: dict):
    """source: {hf: name, config: cfg, split: s, text_field: f} OR {jsonl: path}."""
    if "jsonl" in source:
        import json
        with open(source["jsonl"]) as fh:
            for line in fh:
                yield json.loads(line)[source.get("text_field", "text")]
    else:
        ds = load_dataset(
            source["hf"], source.get("config"),
            split=source.get("split", "train"), streaming=True,
        )
        field = source.get("text_field", "text")
        for row in ds:
            yield row[field]


def make_pairs(
    tokenizer, source: dict, n: int, is_member: int,
    prefix_len: int = 150, suffix_len: int = 50, min_words: int = 220,
) -> Iterator[Pair]:
    """Yield up to `n` (prefix, suffix) pairs, one per document."""
    made = 0
    for i, text in enumerate(_stream(source)):
        if len(text.split()) < min_words:
            continue
        ids = tokenizer(text, add_special_tokens=False).input_ids
        if len(ids) < prefix_len + suffix_len:
            continue
        p_ids = ids[:prefix_len]
        s_ids = ids[prefix_len : prefix_len + suffix_len]
        yield Pair(
            doc_id=f"{source.get('hf', source.get('jsonl'))}#{i}",
            prefix_ids=p_ids, suffix_ids=s_ids,
            prefix_text=tokenizer.decode(p_ids),
            suffix_text=tokenizer.decode(s_ids),
            is_member=is_member,
        )
        made += 1
        if made >= n:
            return
