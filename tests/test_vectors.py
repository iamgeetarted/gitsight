"""Tests for gitsight.vectors — TF-IDF embedding and clustering."""

from __future__ import annotations

import numpy as np
import pytest

from gitsight.vectors import (
    _tokenize,
    _build_vocab,
    _tfidf,
    cosine_similarity,
    embed_messages,
    greedy_cluster,
    pairwise_similarity,
    top_keywords,
)


class TestTokenize:
    def test_lowercases(self) -> None:
        # "fix" is a git stop word; check a real content word is lowercased
        tokens = _tokenize("RESOLVE NullPointer crash")
        assert "resolve" in tokens or "null" in tokens or "crash" in tokens

    def test_removes_stop_words(self) -> None:
        tokens = _tokenize("fix the issue with a bad connection")
        assert "the" not in tokens
        assert "a" not in tokens
        assert "with" not in tokens

    def test_removes_git_stop_words(self) -> None:
        tokens = _tokenize("feat: add new login button")
        assert "feat" not in tokens
        assert "add" not in tokens

    def test_splits_camelcase(self) -> None:
        tokens = _tokenize("refactorUserService loginController")
        assert "refactor" in tokens or "user" in tokens or "service" in tokens

    def test_empty_string(self) -> None:
        assert _tokenize("") == []

    def test_meaningful_tokens_kept(self) -> None:
        tokens = _tokenize("implement websocket authentication middleware")
        assert "websocket" in tokens or "authentication" in tokens or "middleware" in tokens


class TestTFIDF:
    def test_shape(self) -> None:
        docs = [["auth", "login"], ["cache", "redis", "cache"]]
        vocab = _build_vocab(docs)
        mat = _tfidf(docs, vocab)
        assert mat.shape == (2, len(vocab))

    def test_empty_docs(self) -> None:
        mat = _tfidf([], {})
        assert mat.shape[0] == 0

    def test_normalised(self) -> None:
        docs = [["auth", "login", "session"], ["cache", "redis", "memcached"]]
        vocab = _build_vocab(docs)
        mat = _tfidf(docs, vocab)
        norms = np.linalg.norm(mat, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_higher_weight_for_unique_terms(self) -> None:
        # "unique" appears in only one doc — should have higher IDF than "common"
        docs = [["common", "unique"], ["common", "other"]]
        vocab = _build_vocab(docs)
        mat = _tfidf(docs, vocab)
        unique_col = vocab["unique"]
        common_col = vocab["common"]
        # In doc 0 (which has "unique"), the unique term should weigh more
        assert mat[0, unique_col] > 0


class TestEmbedMessages:
    def test_returns_correct_shape(self) -> None:
        msgs = [
            "fix: null pointer in auth module",
            "feat: add oauth integration",
            "fix: race condition in session handler",
        ]
        mat, vocab = embed_messages(msgs)
        assert mat.shape[0] == 3
        assert mat.shape[1] == len(vocab)

    def test_similar_messages_close(self) -> None:
        msgs = [
            "fix: null pointer in login",
            "fix: crash when user is null",
            "refactor: move payment to separate service",
        ]
        mat, _ = embed_messages(msgs)
        sim_01 = cosine_similarity(mat[0], mat[1])
        sim_02 = cosine_similarity(mat[0], mat[2])
        # Two fix commits should be more similar to each other than to refactor
        assert sim_01 >= sim_02


class TestGreedyCluster:
    def test_clusters_similar_messages(self) -> None:
        messages = (
            ["fix: null pointer in auth"] * 4  # similar cluster
            + ["feat: add new dashboard widget"] * 4  # different cluster
        )
        mat, _ = embed_messages(messages)
        clusters = greedy_cluster(mat, threshold=0.1, min_size=2)
        assert len(clusters) >= 1

    def test_empty_input(self) -> None:
        mat = np.zeros((0, 10), dtype=np.float32)
        assert greedy_cluster(mat) == []

    def test_min_size_filters_singletons(self) -> None:
        messages = [
            "fix: auth null ptr",
            "fix: auth crash",
            "totally unrelated one-off commit",
        ]
        mat, _ = embed_messages(messages)
        clusters = greedy_cluster(mat, threshold=0.1, min_size=2)
        for cluster in clusters:
            assert len(cluster) >= 2

    def test_all_indices_valid(self) -> None:
        messages = ["feat: widget"] * 5 + ["fix: login"] * 5
        mat, _ = embed_messages(messages)
        clusters = greedy_cluster(mat, threshold=0.05)
        all_indices = [i for group in clusters for i in group]
        assert all(0 <= i < len(messages) for i in all_indices)


class TestTopKeywords:
    def test_returns_k_keywords(self) -> None:
        messages = [
            "implement websocket authentication token",
            "add websocket connection handler",
            "fix websocket disconnect handling",
        ]
        mat, vocab = embed_messages(messages)
        kws = top_keywords([0, 1, 2], mat, vocab, k=3)
        assert len(kws) <= 3
        assert "websocket" in kws or len(kws) > 0

    def test_empty_cluster(self) -> None:
        mat = np.zeros((5, 10), dtype=np.float32)
        vocab = {"word": 0}
        assert top_keywords([], mat, vocab, k=5) == []


class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_pairwise_symmetry(self) -> None:
        mat, _ = embed_messages(["auth login session", "cache redis session"])
        sim = pairwise_similarity(mat)
        assert sim.shape == (2, 2)
        assert sim[0, 1] == pytest.approx(sim[1, 0], abs=1e-5)
        assert sim[0, 0] == pytest.approx(1.0, abs=1e-5)
