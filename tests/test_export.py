"""Tests for gitsight.export — JSON and Markdown serialisation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from gitsight.analysis import ThemeLabel
from gitsight.export import to_json, to_markdown
from gitsight.git import Commit


def _make_commit(
    sha: str = "abc12345def6789a",
    author: str = "Alice",
    subject: str = "feat: add feature",
    date: str = "2026-01-15",
) -> Commit:
    ts = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    return Commit(sha=sha, author=author, email="a@b.com", timestamp=ts, subject=subject, body="")


def _make_label(
    title: str = "Auth Refactoring",
    summary: str = "Moved auth to its own module.",
    action_items: str = "Review token expiry logic.",
) -> ThemeLabel:
    return ThemeLabel(title=title, summary=summary, action_items=action_items)


@pytest.fixture
def sample_data() -> tuple[list[Commit], list[list[int]], list[ThemeLabel], Path]:
    commits = [
        _make_commit("aaaabbbbcccc0001", subject="feat: add OAuth"),
        _make_commit("aaaabbbbcccc0002", subject="fix: token expiry"),
        _make_commit("aaaabbbbcccc0003", subject="feat: add dashboard"),
        _make_commit("aaaabbbbcccc0004", subject="chore: update deps"),
    ]
    clusters = [[0, 1], [2]]
    labels = [_make_label("Auth Work"), _make_label("Frontend Work")]
    repo = Path("/tmp/my-repo")
    return commits, clusters, labels, repo


class TestToJson:
    def test_valid_json(self, sample_data) -> None:
        commits, clusters, labels, repo = sample_data
        result = to_json(commits, clusters, labels, repo)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_top_level_keys(self, sample_data) -> None:
        commits, clusters, labels, repo = sample_data
        parsed = json.loads(to_json(commits, clusters, labels, repo))
        assert "generated_at" in parsed
        assert "total_commits" in parsed
        assert "themes" in parsed
        assert "themes_found" in parsed

    def test_total_commits_count(self, sample_data) -> None:
        commits, clusters, labels, repo = sample_data
        parsed = json.loads(to_json(commits, clusters, labels, repo))
        assert parsed["total_commits"] == 4

    def test_themes_structure(self, sample_data) -> None:
        commits, clusters, labels, repo = sample_data
        parsed = json.loads(to_json(commits, clusters, labels, repo))
        theme = parsed["themes"][0]
        assert "title" in theme
        assert "summary" in theme
        assert "commit_count" in theme
        assert "commits" in theme
        assert "date_range" in theme
        assert "authors" in theme

    def test_cluster_commit_count(self, sample_data) -> None:
        commits, clusters, labels, repo = sample_data
        parsed = json.loads(to_json(commits, clusters, labels, repo))
        assert parsed["themes"][0]["commit_count"] == 2
        assert parsed["themes"][1]["commit_count"] == 1

    def test_overall_summary_included(self, sample_data) -> None:
        commits, clusters, labels, repo = sample_data
        parsed = json.loads(to_json(commits, clusters, labels, repo, overall_summary="Great progress."))
        assert parsed["overall_summary"] == "Great progress."

    def test_sha_truncated_to_8(self, sample_data) -> None:
        commits, clusters, labels, repo = sample_data
        parsed = json.loads(to_json(commits, clusters, labels, repo))
        sha = parsed["themes"][0]["commits"][0]["sha"]
        assert len(sha) == 8


class TestToMarkdown:
    def test_returns_string(self, sample_data) -> None:
        commits, clusters, labels, repo = sample_data
        result = to_markdown(commits, clusters, labels, repo)
        assert isinstance(result, str)
        assert len(result) > 100

    def test_has_title(self, sample_data) -> None:
        commits, clusters, labels, repo = sample_data
        result = to_markdown(commits, clusters, labels, repo)
        assert "gitsight Report" in result

    def test_theme_titles_present(self, sample_data) -> None:
        commits, clusters, labels, repo = sample_data
        result = to_markdown(commits, clusters, labels, repo)
        assert "Auth Work" in result
        assert "Frontend Work" in result

    def test_overall_summary_included(self, sample_data) -> None:
        commits, clusters, labels, repo = sample_data
        result = to_markdown(commits, clusters, labels, repo, overall_summary="Overall great work.")
        assert "Executive Summary" in result
        assert "Overall great work." in result

    def test_unclustered_section(self, sample_data) -> None:
        commits, clusters, labels, repo = sample_data
        # commit index 3 is not in any cluster
        result = to_markdown(commits, clusters, labels, repo)
        assert "Unclustered" in result

    def test_action_items_included(self, sample_data) -> None:
        commits, clusters, labels, repo = sample_data
        result = to_markdown(commits, clusters, labels, repo)
        assert "Review token expiry" in result

    def test_no_unclustered_when_all_covered(self) -> None:
        commits = [_make_commit("a1"), _make_commit("a2")]
        clusters = [[0, 1]]
        labels = [_make_label()]
        result = to_markdown(commits, clusters, labels, Path("/repo"))
        assert "Unclustered" not in result
