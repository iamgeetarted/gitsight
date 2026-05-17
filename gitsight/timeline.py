"""Commit activity timeline — weekly bar chart rendered with Rich."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .git import Commit


def _week_start(dt: datetime) -> str:
    """Return the ISO date of the Monday that begins dt's week."""
    monday = dt - timedelta(days=dt.weekday())
    return monday.strftime("%Y-%m-%d")


def print_timeline(commits: "list[Commit]", max_weeks: int = 26) -> None:
    """Render a weekly commit-frequency bar chart to the terminal.

    Groups commits by calendar week and draws an ASCII bar for each week,
    scaled to the busiest week. Limited to the most recent *max_weeks* weeks
    so the chart stays readable in a standard terminal.
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

    weeks: Counter[str] = Counter()
    for c in commits:
        try:
            dt = datetime.strptime(c.date_str, "%Y-%m-%d")
        except ValueError:
            continue
        weeks[_week_start(dt)] += 1

    if not weeks:
        return

    sorted_weeks = sorted(weeks.items())[-max_weeks:]
    max_count = max(v for _, v in sorted_weeks)
    bar_width = 32

    console.print()
    console.print(Rule("[bold]Commit Timeline[/bold]  [dim](weekly)[/dim]", style="dim cyan"))

    t = Table(
        box=rbox.SIMPLE,
        show_header=False,
        pad_edge=False,
        padding=(0, 1),
    )
    t.add_column("Week", style="dim", no_wrap=True)
    t.add_column("Bar", no_wrap=True)
    t.add_column("N", justify="right", style="cyan")

    for week_str, count in sorted_weeks:
        filled = max(1, int(round(count / max_count * bar_width))) if count else 0
        bar = (
            "[bold green]" + "█" * filled + "[/bold green]"
            + "[dim]" + "░" * (bar_width - filled) + "[/dim]"
        )
        t.add_row(week_str, bar, str(count))

    console.print(t)
    console.print()
