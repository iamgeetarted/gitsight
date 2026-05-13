"""Lightweight vector similarity — pure numpy, zero heavy ML dependencies.

Converts commit messages to bag-of-words TF-IDF vectors and groups them
by cosine similarity.  No sentence-transformers, no API calls, no FAISS.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Text → vector
# ---------------------------------------------------------------------------

_STOP = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would could should may might shall can need must "
    "this that these those i we you he she it they "
    "to of in on at by for with from as into and or but not no "
    "s t re ve ll d m".split()
)

# Common git noise words that carry little semantic signal
_GIT_STOP = frozenset(
    "feat fix chore docs style refactor test perf ci build revert merge "
    "update add remove change bump version release minor major patch "
    "pr pull request branch commit initial first".split()
)

_STOP_ALL = _STOP | _GIT_STOP


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace and CamelCase."""
    # CamelCase split must happen before lowercasing
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = text.lower()
    # Split on non-alphanumeric characters
    tokens = re.findall(r"[a-z][a-z0-9]*", text)
    return [t for t in tokens if len(t) > 1 and t not in _STOP_ALL]


def _build_vocab(documents: list[list[str]]) -> dict[str, int]:
    """Build a vocabulary mapping token → index from a list of token lists."""
    vocab: dict[str, int] = {}
    for tokens in documents:
        for t in tokens:
            if t not in vocab:
                vocab[t] = len(vocab)
    return vocab


def _tfidf(
    documents: list[list[str]],
    vocab: dict[str, int],
) -> np.ndarray:
    """Compute a (n_docs, vocab_size) TF-IDF matrix."""
    n = len(documents)
    v = len(vocab)
    if n == 0 or v == 0:
        return np.zeros((n, max(v, 1)), dtype=np.float32)

    # Term-frequency matrix
    tf = np.zeros((n, v), dtype=np.float32)
    for i, tokens in enumerate(documents):
        if not tokens:
            continue
        counts = Counter(tokens)
        total = sum(counts.values())
        for token, cnt in counts.items():
            if token in vocab:
                tf[i, vocab[token]] = cnt / total

    # Document frequency
    df = (tf > 0).sum(axis=0)
    idf = np.log((1 + n) / (1 + df)) + 1.0  # sklearn-style smooth IDF

    mat = tf * idf

    # L2-normalise each row so cosine = dot product
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (mat / norms).astype(np.float32)


def embed_messages(messages: Sequence[str]) -> tuple[np.ndarray, dict[str, int]]:
    """Embed commit messages into a TF-IDF matrix.

    Args:
        messages: Commit subjects or full messages.

    Returns:
        Tuple of (matrix of shape (n, vocab_size), vocab dict).
    """
    token_lists = [_tokenize(m) for m in messages]
    vocab = _build_vocab(token_lists)
    matrix = _tfidf(token_lists, vocab)
    return matrix, vocab


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalised vectors."""
    return float(np.dot(a, b))


def pairwise_similarity(matrix: np.ndarray) -> np.ndarray:
    """Return an (n, n) cosine similarity matrix for a normalised matrix."""
    return (matrix @ matrix.T).astype(np.float32)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def greedy_cluster(
    matrix: np.ndarray,
    threshold: float = 0.25,
    min_size: int = 2,
) -> list[list[int]]:
    """Group row indices by cosine similarity using greedy single-pass clustering.

    Each unassigned commit seeds a new cluster; subsequent commits join the
    nearest existing cluster if similarity exceeds *threshold*.

    Args:
        matrix: L2-normalised (n, vocab_size) TF-IDF matrix.
        threshold: Minimum cosine similarity to join an existing cluster.
        min_size: Discard clusters smaller than this.

    Returns:
        List of index-lists, largest clusters first.
    """
    n = matrix.shape[0]
    if n == 0:
        return []

    assigned = np.full(n, -1, dtype=np.int32)
    centroids: list[np.ndarray] = []
    cluster_members: list[list[int]] = []

    for i in range(n):
        vec = matrix[i]
        best_cluster = -1
        best_sim = threshold

        for cid, centroid in enumerate(centroids):
            sim = cosine_similarity(vec, centroid)
            if sim > best_sim:
                best_sim = sim
                best_cluster = cid

        if best_cluster == -1:
            # New cluster
            centroids.append(vec.copy())
            cluster_members.append([i])
            assigned[i] = len(cluster_members) - 1
        else:
            cluster_members[best_cluster].append(i)
            assigned[i] = best_cluster
            # Update centroid (running mean, re-normalised)
            members = cluster_members[best_cluster]
            c = matrix[members].mean(axis=0)
            norm = np.linalg.norm(c)
            centroids[best_cluster] = c / norm if norm > 0 else c

    # Filter small clusters, sort by size desc
    result = [m for m in cluster_members if len(m) >= min_size]
    result.sort(key=len, reverse=True)
    return result


def top_keywords(
    cluster_indices: list[int],
    matrix: np.ndarray,
    vocab: dict[str, int],
    k: int = 5,
) -> list[str]:
    """Return the *k* highest-TF-IDF keywords for a cluster.

    Args:
        cluster_indices: Row indices belonging to the cluster.
        matrix: The TF-IDF matrix (NOT necessarily L2-normalised here).
        vocab: Token-to-column mapping.
        k: Number of keywords to return.
    """
    if not cluster_indices:
        return []
    centroid = matrix[cluster_indices].mean(axis=0)
    idx2word = {v: k for k, v in vocab.items()}
    top = np.argsort(centroid)[::-1][:k]
    return [idx2word[i] for i in top if i in idx2word and centroid[i] > 0]
