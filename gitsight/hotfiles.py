"""File churn analysis — which files change most often across git history."""

from __future__ import annotations

import asyncio
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FileChurnEntry:
    """Aggregated churn statistics for a single file path."""

    path: str
    commit_count: int
    lines_added: int
    lines_deleted: int

    @property
    def churn_score(self) -> int:
        """Sum of all line additions and deletions — a raw activity proxy."""
        return self.lines_added + self.lines_deleted

    @property
    def net_change(self) -> int:
        return self.lines_added - self.lines_deleted


async def read_file_churn_async(
    repo_path: Path,
    max_count: int = 500,
    since: str | None = None,
    until: str | None = None,
    author: str | None = None,
    branch: str = "HEAD",
    min_commits: int = 2,
) -> list[FileChurnEntry]:
    """Parse `git log --numstat` and return per-file churn aggregates.

    Args:
        repo_path: Path to the git repository root.
        max_count: Maximum number of commits to scan.
        since: Optional date lower bound (e.g. "3 months ago").
        until: Optional date upper bound.
        author: Optional author filter.
        branch: Ref to read from.
        min_commits: Exclude files touched fewer times than this threshold.

    Returns:
        List of FileChurnEntry sorted by commit_count descending.
    """
    cmd = [
        "git", "-C", str(repo_path),
        "log",
        "--numstat",
        "--pretty=format:COMMIT",
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
            f"git log --numstat failed: {stderr.decode().strip()}"
        )

    commit_counts: dict[str, int] = {}
    lines_added: dict[str, int] = {}
    lines_deleted: dict[str, int] = {}

    for line in stdout.decode().splitlines():
        line = line.strip()
        if not line or line == "COMMIT":
            continue
        parts = line.split("\t", 2)
        if len(parts) != 3:
            continue
        added_str, deleted_str, path = parts
        # Binary files show "-" instead of a number
        if added_str == "-" or deleted_str == "-":
            continue
        try:
            added = int(added_str)
            deleted = int(deleted_str)
        except ValueError:
            continue

        commit_counts[path] = commit_counts.get(path, 0) + 1
        lines_added[path] = lines_added.get(path, 0) + added
        lines_deleted[path] = lines_deleted.get(path, 0) + deleted

    entries = [
        FileChurnEntry(
            path=path,
            commit_count=count,
            lines_added=lines_added.get(path, 0),
            lines_deleted=lines_deleted.get(path, 0),
        )
        for path, count in commit_counts.items()
        if count >= min_commits
    ]
    entries.sort(key=lambda e: (e.commit_count, e.churn_score), reverse=True)
    return entries


def render_hotfiles(
    entries: list[FileChurnEntry],
    top_n: int = 20,
) -> None:
    """Render a Rich heatmap table of the hottest files.

    Args:
        entries: Sorted FileChurnEntry list (highest churn first).
        top_n: Maximum rows to display.
    """
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.table import Table
    from rich.text import Text

    console = Console()
    shown = entries[:top_n]

    if not shown:
        console.print("[dim]No files found with sufficient commit history.[/dim]")
        return

    max_commits = shown[0].commit_count

    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
        pad_edge=True,
    )
    table.add_column("#", style="dim", width=3, justify="right")
    table.add_column("File", style="bold white", no_wrap=False)
    table.add_column("Commits", justify="right", width=8)
    table.add_column("Heat", width=22)
    table.add_column("+Lines", justify="right", style="green", width=7)
    table.add_column("-Lines", justify="right", style="red", width=7)

    bar_width = 18
    for i, entry in enumerate(shown, 1):
        filled = max(1, int(round(entry.commit_count / max_commits * bar_width)))
        intensity = entry.commit_count / max_commits
        if intensity > 0.7:
            bar_color = "bold red"
        elif intensity > 0.4:
            bar_color = "yellow"
        else:
            bar_color = "green"

        bar = (
            f"[{bar_color}]" + "█" * filled + f"[/{bar_color}]"
            + "[dim]" + "░" * (bar_width - filled) + "[/dim]"
        )

        # Shorten deeply nested paths for readability
        display_path = entry.path
        parts = Path(entry.path).parts
        if len(parts) > 4:
            display_path = str(Path(*parts[:1]) / "…" / Path(*parts[-2:]))

        table.add_row(
            str(i),
            display_path,
            str(entry.commit_count),
            bar,
            f"+{entry.lines_added:,}",
            f"-{entry.lines_deleted:,}",
        )

    console.print()
    console.print(Rule("[bold]File Churn Heatmap[/bold]  [dim](most-changed files)[/dim]", style="dim cyan"))
    console.print(Panel(table, border_style="cyan", box=box.ROUNDED))
    console.print(
        f"[dim]Showing top {len(shown)} of {len(entries)} files "
        f"with ≥2 commits. Sort: commit count.[/dim]"
    )
    console.print()
