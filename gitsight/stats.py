"""Per-author commit velocity statistics."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .git import Commit


def print_author_stats(commits: "list[Commit]") -> None:
    """Render a Rich table of per-author commit counts, date ranges, and weekly cadence.

    For each author, computes:
    - Total commit count
    - First and last commit date
    - Active span in days
    - Average commits per week over that span
    """
    try:
        from rich.console import Console
        from rich.rule import Rule
        from rich.table import Table
        from rich.table import box as rbox
    except ImportError:
        return

    if not commits:
        return

    console = Console()

    by_author: dict[str, list[Commit]] = defaultdict(list)
    for c in commits:
        by_author[c.author].append(c)

    sorted_authors = sorted(by_author.items(), key=lambda x: -len(x[1]))

    console.print(Rule("[bold]Author Velocity[/bold]", style="dim cyan"))

    t = Table(
        box=rbox.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
        pad_edge=True,
    )
    t.add_column("Author", style="bold white", no_wrap=True, min_width=16)
    t.add_column("Commits", justify="right", style="cyan")
    t.add_column("First commit", style="dim", no_wrap=True)
    t.add_column("Last commit", style="dim", no_wrap=True)
    t.add_column("Span (d)", justify="right", style="dim")
    t.add_column("Avg / wk", justify="right", style="bold green")

    for author, clist in sorted_authors:
        dates: list[datetime] = []
        for c in clist:
            try:
                dates.append(datetime.strptime(c.date_str, "%Y-%m-%d"))
            except ValueError:
                pass

        if dates:
            first_str = min(dates).strftime("%Y-%m-%d")
            last_str = max(dates).strftime("%Y-%m-%d")
            span_days = max(1, (max(dates) - min(dates)).days + 1)
            avg_week = f"{len(clist) / (span_days / 7):.1f}"
        else:
            first_str = last_str = "—"
            span_days = 0
            avg_week = "—"

        t.add_row(
            author,
            str(len(clist)),
            first_str,
            last_str,
            str(span_days) if span_days else "—",
            avg_week,
        )

    console.print(t)
    console.print()
