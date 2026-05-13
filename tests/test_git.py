"""Tests for gitsight.git — commit parsing and grouping."""

from __future__ import annotations

import subprocess
import tempfile
from datetime import timezone
from pathlib import Path

import pytest

from gitsight.git import (
    Commit,
    commits_by_date,
    get_authors,
    read_commits,
    _parse_raw,
)

_SEP = "\x1f"
_REC = "\x1e"


def _make_record(
    sha: str = "abc123def456",
    author: str = "Alice",
    email: str = "alice@example.com",
    ts: str = "1700000000",
    subject: str = "fix: resolve null pointer",
    body: str = "",
) -> str:
    return f"{sha}{_SEP}{author}{_SEP}{email}{_SEP}{ts}{_SEP}{subject}{_SEP}{body}{_REC}"


class TestParseRaw:
    def test_single_commit(self) -> None:
        raw = _make_record(subject="feat: add login flow")
        commits = _parse_raw(raw)
        assert len(commits) == 1
        assert commits[0].subject == "feat: add login flow"
        assert commits[0].author == "Alice"

    def test_multiple_commits(self) -> None:
        raw = _make_record(subject="fix: null ptr") + _make_record(subject="docs: update README")
        commits = _parse_raw(raw)
        assert len(commits) == 2
        assert commits[0].subject == "fix: null ptr"
        assert commits[1].subject == "docs: update README"

    def test_empty_raw(self) -> None:
        assert _parse_raw("") == []
        assert _parse_raw("   \n  ") == []

    def test_invalid_record_skipped(self) -> None:
        raw = f"bad record{_REC}" + _make_record(subject="good commit")
        commits = _parse_raw(raw)
        # "bad record" has < 5 fields, should be skipped
        assert len(commits) == 1
        assert commits[0].subject == "good commit"

    def test_timestamp_parsed(self) -> None:
        raw = _make_record(ts="1700000000")
        commits = _parse_raw(raw)
        assert commits[0].timestamp.tzinfo == timezone.utc
        assert commits[0].timestamp.year == 2023

    def test_message_property(self) -> None:
        raw = _make_record(subject="fix: crash", body="Fixes a crash when X is None.")
        commit = _parse_raw(raw)[0]
        assert "fix: crash" in commit.message
        assert "Fixes a crash" in commit.message


class TestGetAuthors:
    def test_counts_correctly(self) -> None:
        raw = (
            _make_record(author="Alice")
            + _make_record(author="Bob")
            + _make_record(author="Alice")
        )
        commits = _parse_raw(raw)
        authors = get_authors(commits)
        assert authors["Alice"] == 2
        assert authors["Bob"] == 1

    def test_sorted_by_count_desc(self) -> None:
        raw = (
            _make_record(author="Bob")
            + _make_record(author="Alice")
            + _make_record(author="Alice")
        )
        commits = _parse_raw(raw)
        names = list(get_authors(commits).keys())
        assert names[0] == "Alice"


class TestCommitsByDate:
    def test_groups_by_date(self) -> None:
        raw = (
            _make_record(ts="1700000000")  # 2023-11-14
            + _make_record(ts="1700086400")  # 2023-11-15 (+ 1 day roughly)
        )
        commits = _parse_raw(raw)
        by_date = commits_by_date(commits)
        assert len(by_date) >= 1
        # All dates should be present as keys
        for date_str in by_date:
            assert len(date_str) == 10  # YYYY-MM-DD

    def test_sorted_chronologically(self) -> None:
        raw = (
            _make_record(ts="1700086400")   # later
            + _make_record(ts="1700000000")  # earlier
        )
        commits = _parse_raw(raw)
        by_date = commits_by_date(commits)
        keys = list(by_date.keys())
        assert keys == sorted(keys)


@pytest.fixture
def temp_git_repo(tmp_path: Path) -> Path:
    """Create a minimal git repo with a few commits."""
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "test@test.com"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "Test User"], check=True, capture_output=True)

    for i, msg in enumerate(["feat: add auth module", "fix: login crash", "docs: update README"]):
        (tmp_path / f"file{i}.txt").write_text(msg)
        subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True, capture_output=True)
        subprocess.run(
            ["git", "-C", str(tmp_path), "commit", "-m", msg, "--allow-empty"],
            check=True,
            capture_output=True,
            env={"GIT_AUTHOR_NAME": "Test", "GIT_AUTHOR_EMAIL": "t@t.com",
                 "GIT_COMMITTER_NAME": "Test", "GIT_COMMITTER_EMAIL": "t@t.com",
                 "PATH": "/usr/bin:/bin"},
        )

    return tmp_path


class TestReadCommits:
    def test_reads_commits(self, temp_git_repo: Path) -> None:
        commits = read_commits(temp_git_repo, max_count=10)
        assert len(commits) == 3

    def test_subjects_present(self, temp_git_repo: Path) -> None:
        commits = read_commits(temp_git_repo)
        subjects = [c.subject for c in commits]
        assert "docs: update README" in subjects

    def test_bad_repo_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            read_commits(tmp_path / "nonexistent")
