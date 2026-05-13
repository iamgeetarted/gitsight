"""Git log parsing — read commit history from a local repository."""

from __future__ import annotations

import asyncio
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class Commit:
    """Immutable snapshot of a single git commit."""

    sha: str
    author: str
    email: str
    timestamp: datetime
    subject: str
    body: str

    @property
    def message(self) -> str:
        """Full commit message (subject + body)."""
        return f"{self.subject}\n\n{self.body}".strip()

    @property
    def date_str(self) -> str:
        return self.timestamp.strftime("%Y-%m-%d")


_SEP = "\x1f"   # unit separator — safe delimiter inside git log output
_REC = "\x1e"   # record separator between commits


def _parse_raw(raw: str) -> list[Commit]:
    """Parse git log --format output into Commit objects."""
    commits: list[Commit] = []
    for record in raw.split(_REC):
        record = record.strip()
        if not record:
            continue
        parts = record.split(_SEP, 5)
        if len(parts) < 5:
            continue
        sha, author, email, ts_str, subject, *rest = parts
        body = rest[0].strip() if rest else ""
        try:
            ts = datetime.fromtimestamp(int(ts_str), tz=timezone.utc)
        except (ValueError, OSError):
            ts = datetime.now(timezone.utc)
        commits.append(Commit(
            sha=sha.strip(),
            author=author.strip(),
            email=email.strip(),
            timestamp=ts,
            subject=subject.strip(),
            body=body,
        ))
    return commits


def read_commits(
    repo_path: Path,
    max_count: int = 200,
    since: str | None = None,
    until: str | None = None,
    author: str | None = None,
    branch: str = "HEAD",
) -> list[Commit]:
    """Read git log from *repo_path* and return parsed Commit objects.

    Args:
        repo_path: Path to the git repository root.
        max_count: Maximum number of commits to return.
        since: Optionally limit to commits after this date (e.g. "2 weeks ago").
        until: Optionally limit to commits before this date.
        author: Filter by author name or email pattern.
        branch: Branch or ref to read from (default: HEAD).

    Returns:
        List of Commit objects, newest first.

    Raises:
        ValueError: If the path is not a git repository or git is unavailable.
    """
    fmt = f"%H{_SEP}%an{_SEP}%ae{_SEP}%ct{_SEP}%s{_SEP}%b{_REC}"

    cmd = [
        "git", "-C", str(repo_path),
        "log",
        f"--format={fmt}",
        f"--max-count={max_count}",
        branch,
    ]
    if since:
        cmd.append(f"--since={since}")
    if until:
        cmd.append(f"--until={until}")
    if author:
        cmd.append(f"--author={author}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except subprocess.CalledProcessError as exc:
        raise ValueError(
            f"git log failed in {repo_path}: {exc.stderr.strip()}"
        ) from exc
    except FileNotFoundError as exc:
        raise ValueError("git is not installed or not on PATH") from exc

    return _parse_raw(result.stdout)


async def read_commits_async(
    repo_path: Path,
    max_count: int = 200,
    since: str | None = None,
    until: str | None = None,
    author: str | None = None,
    branch: str = "HEAD",
) -> list[Commit]:
    """Async version of :func:`read_commits` using asyncio subprocess.

    Args:
        repo_path: Path to the git repository root.
        max_count: Maximum commits to return.
        since: Optional date filter (e.g. "2 weeks ago").
        until: Optional upper date bound.
        author: Optional author filter.
        branch: Ref to read from.

    Returns:
        List of Commit objects, newest first.
    """
    fmt = f"%H{_SEP}%an{_SEP}%ae{_SEP}%ct{_SEP}%s{_SEP}%b{_REC}"

    cmd = [
        "git", "-C", str(repo_path),
        "log",
        f"--format={fmt}",
        f"--max-count={max_count}",
        branch,
    ]
    if since:
        cmd.append(f"--since={since}")
    if until:
        cmd.append(f"--until={until}")
    if author:
        cmd.append(f"--author={author}")

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise ValueError(
            f"git log failed in {repo_path}: {stderr.decode().strip()}"
        )
    return _parse_raw(stdout.decode())


def get_authors(commits: list[Commit]) -> dict[str, int]:
    """Return a {author_name: commit_count} mapping, sorted by count desc."""
    counts: dict[str, int] = {}
    for c in commits:
        counts[c.author] = counts.get(c.author, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


def commits_by_date(commits: list[Commit]) -> dict[str, list[Commit]]:
    """Group commits by date string (YYYY-MM-DD), sorted chronologically."""
    groups: dict[str, list[Commit]] = {}
    for c in commits:
        groups.setdefault(c.date_str, []).append(c)
    return dict(sorted(groups.items()))
