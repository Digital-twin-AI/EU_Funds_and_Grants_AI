"""Lightweight Okapi BM25 + RRF for hybrid grant search."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

_TOKEN_RE = re.compile(r"[a-zA-ZšđčćžŠĐČĆŽ0-9]+", re.UNICODE)
_STOPWORDS = frozenset({
    "i", "u", "na", "za", "od", "do", "je", "su", "se", "da", "a", "o",
    "the", "and", "or", "for", "of", "in", "to", "an", "with",
    "grant", "grants", "fond", "fondovi", "program", "poziv",
})


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
    return [t for t in tokens if len(t) > 1 and t not in _STOPWORDS]


@dataclass
class BM25Index:
    k1: float = 1.5
    b: float = 0.75
    ids: list[str] = field(default_factory=list)
    doc_len: list[int] = field(default_factory=list)
    doc_freqs: list[Counter] = field(default_factory=list)
    df: dict[str, int] = field(default_factory=dict)
    avgdl: float = 0.0
    n_docs: int = 0

    def build(self, ids: list[str], documents: list[str]) -> None:
        if len(ids) != len(documents):
            raise ValueError("ids and documents length mismatch")
        self.ids = list(ids)
        self.doc_freqs = []
        self.doc_len = []
        df_acc: dict[str, int] = defaultdict(int)
        for doc in documents:
            tokens = tokenize(doc)
            freqs = Counter(tokens)
            self.doc_freqs.append(freqs)
            self.doc_len.append(len(tokens))
            for term in freqs:
                df_acc[term] += 1
        self.n_docs = len(ids)
        self.avgdl = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0
        self.df = dict(df_acc)

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))

    def score(self, query: str) -> list[tuple[str, float]]:
        q_tokens = tokenize(query)
        if not q_tokens or self.n_docs == 0:
            return [(doc_id, 0.0) for doc_id in self.ids]
        scores: list[tuple[str, float]] = []
        for idx, doc_id in enumerate(self.ids):
            freqs = self.doc_freqs[idx]
            dl = self.doc_len[idx]
            score = 0.0
            for term in q_tokens:
                if term not in freqs:
                    continue
                tf = freqs[term]
                idf = self._idf(term)
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / (self.avgdl or 1.0))
                score += idf * (tf * (self.k1 + 1.0)) / (denom or 1.0)
            scores.append((doc_id, score))
        return scores

    def top_n(self, query: str, n: int) -> list[tuple[str, float]]:
        ranked = sorted(self.score(query), key=lambda x: x[1], reverse=True)
        return [(doc_id, s) for doc_id, s in ranked[: max(n, 0)] if s > 0]


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    *,
    k: int = 60,
    weights: list[float] | None = None,
) -> list[tuple[str, float]]:
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights length must match ranked_lists")
    scores: dict[str, float] = defaultdict(float)
    for weight, ranking in zip(weights, ranked_lists):
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += weight * (1.0 / (k + rank))
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
