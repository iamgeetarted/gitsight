"""Rich Live dashboard — streams analysis progress and results in real-time."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich import box as rich_box

if TYPE_CHECKING:
    from .git import Commit
    from .analysis import ThemeLabel


console = Console()


@dataclass
class AnalysisState:
    """Mutable state driving the live dashboard."""

    total_commits: int = 0
    clusters_found: int = 0
    labels_done: int = 0
    labels_total: int = 0
    phase: str = "Initializing…"
    themes: list[tuple[str, int, str]] = field(default_factory=list)  # (title, count, status)
    current_stream: str = ""


def _theme_table(state: AnalysisState) -> Table:
    t = Table(
        box=rich_box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        border_style="cyan",
        expand=True,
        title=f"[bold cyan]Themes Discovered[/bold cyan]  "
              f"[dim]({state.labels_done}/{state.labels_total})[/dim]",
    )
    t.add_column("#", style="dim", width=3, justify="right")
    t.add_column("Theme", no_wrap=False)
    t.add_column("Commits", justify="right", width=8, style="yellow")
    t.add_column("Status", width=12)

    for i, (title, count, status) in enumerate(state.themes, 1):
        if status == "done":
            status_text = Text("✓ done", style="bold green")
        elif status == "streaming":
            status_text = Text("⟳ labeling…", style="cyan")
        else:
            status_text = Text("pending", style="dim")
        t.add_row(str(i), title, str(count), status_text)

    return t


def _header_panel(state: AnalysisState) -> Panel:
    grid = Table.grid(expand=True)
    grid.add_column()
    grid.add_column(justify="right")
    grid.add_row(
        f"[bold cyan]gitsight[/bold cyan]  [dim]{state.phase}[/dim]",
        f"[dim]{state.total_commits} commits  ·  {state.clusters_found} clusters[/dim]",
    )
    return Panel(grid, box=rich_box.ROUNDED, border_style="cyan")


def _stream_panel(state: AnalysisState) -> Panel:
    text = state.current_stream[-600:] if state.current_stream else "[dim]Waiting for AI…[/dim]"
    return Panel(
        text,
        title="[bold]Live AI Output[/bold]",
        border_style="dim",
        box=rich_box.ROUNDED,
    )


def make_layout() -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
    )
    layout["body"].split_row(
        Layout(name="themes", ratio=2),
        Layout(name="stream", ratio=1),
    )
    return layout


def print_summary_header(commits: list[Commit]) -> None:
    """Print a static summary before the live phase begins."""
    from collections import Counter
    from .git import get_authors

    console.print()
    console.print(Rule("[bold cyan]gitsight[/bold cyan]"))
    console.print(f"  [bold]{len(commits)}[/bold] commits loaded")

    authors = get_authors(commits)
    if authors:
        top_authors = list(authors.items())[:5]
        author_str = "  ".join(f"[cyan]{a}[/cyan] [dim]{n}[/dim]" for a, n in top_authors)
        console.print(f"  Top authors: {author_str}")

    if commits:
        oldest = commits[-1].date_str
        newest = commits[0].date_str
        console.print(f"  Date range:  [yellow]{oldest}[/yellow] → [yellow]{newest}[/yellow]")

    console.print()


def print_theme_result(
    index: int,
    label: ThemeLabel,
    commits: list[Commit],
    keywords: list[str],
) -> None:
    """Print a finalised theme block after streaming completes."""
    kw_str = ", ".join(keywords[:6]) if keywords else ""

    console.print()
    console.print(
        Rule(
            f"[bold]{index}. {label.title}[/bold]  "
            f"[dim]{len(commits)} commits[/dim]",
            style="cyan",
        )
    )

    if label.summary:
        console.print(f"  {label.summary}")

    if label.action_items and label.action_items.lower() not in ("none", ""):
        console.print(f"  [dim]► Action:[/dim] {label.action_items}")

    if kw_str:
        console.print(f"  [dim]Keywords: {kw_str}[/dim]")

    # Recent commits in this cluster
    t = Table(box=rich_box.SIMPLE, show_header=False, pad_edge=False)
    t.add_column("SHA", style="dim", width=9, no_wrap=True)
    t.add_column("Date", style="dim", width=11, no_wrap=True)
    t.add_column("Author", style="cyan", width=18, no_wrap=True)
    t.add_column("Subject", no_wrap=False)

    for c in commits[:8]:
        t.add_row(c.sha[:8], c.date_str, c.author[:16], c.subject)
    if len(commits) > 8:
        t.add_row("…", "", "", f"[dim]+{len(commits)-8} more[/dim]")

    console.print(t)


def print_overall_summary(text: str) -> None:
    """Print the final executive summary."""
    console.print()
    console.print(Rule("[bold cyan]Executive Summary[/bold cyan]"))
    console.print(text)
    console.print()
