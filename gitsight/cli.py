"""Command-line interface for gitsight."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from . import __version__


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------

async def _run_analysis(args: argparse.Namespace) -> int:
    """Orchestrate the full analysis pipeline with structured concurrency."""
    from .git import read_commits_async
    from .vectors import embed_messages, greedy_cluster, top_keywords
    from .renderer import (
        print_summary_header,
        print_theme_result,
        print_overall_summary,
        console,
    )

    repo = Path(args.repo).resolve()
    if not (repo / ".git").exists() and not (repo / "HEAD").exists():
        print(f"gitsight: not a git repository: {repo}", file=sys.stderr)
        return 2

    # ── Phase 1: read commits ──────────────────────────────────────────────
    try:
        commits = await read_commits_async(
            repo,
            max_count=args.max_commits,
            since=args.since,
            until=args.until,
            author=args.author,
            branch=args.branch,
        )
    except ValueError as exc:
        print(f"gitsight: {exc}", file=sys.stderr)
        return 2

    if not commits:
        print("gitsight: no commits found matching the given filters.")
        return 0

    print_summary_header(commits)

    # ── Phase 1b: optional timeline chart ────────────────────────────────
    if args.timeline:
        from .timeline import print_timeline
        print_timeline(commits)

    # ── Phase 1c: optional author velocity table ──────────────────────────
    if args.author_stats:
        from .stats import print_author_stats
        print_author_stats(commits)

    # ── Phase 2: vector clustering ─────────────────────────────────────────
    messages = [c.message for c in commits]
    matrix, vocab = embed_messages(messages)
    clusters = greedy_cluster(matrix, threshold=args.threshold, min_size=args.min_cluster_size)

    if not clusters:
        console.print("[yellow]No clusters found — try lowering --threshold or --min-cluster-size.[/yellow]")
        if not args.no_ai:
            console.print("[dim]Skipping AI labeling — nothing to label.[/dim]")
        return 0

    console.print(
        f"  [bold]{len(clusters)}[/bold] clusters found  "
        f"[dim](threshold={args.threshold}, min-size={args.min_cluster_size})[/dim]"
    )

    keywords_per_cluster = [
        top_keywords(idx_list, matrix, vocab, k=8)
        for idx_list in clusters
    ]

    # ── Phase 3: AI labeling (concurrent via TaskGroup) ───────────────────
    labels = []

    if args.no_ai:
        from .analysis import ThemeLabel
        for kws in keywords_per_cluster:
            title = " / ".join(kws[:3]).title() if kws else "Uncategorised"
            labels.append(ThemeLabel(title=title, summary="", action_items="None"))
    else:
        try:
            from .analysis import label_clusters_concurrent
        except ImportError as exc:
            print(f"gitsight: {exc}", file=sys.stderr)
            return 1

        clusters_commits = [[commits[i] for i in idx_list] for idx_list in clusters]

        done_count = 0

        def _on_label(idx: int, text: str) -> None:
            nonlocal done_count
            done_count += 1

        console.print()
        console.print(
            f"  Labeling [bold]{len(clusters)}[/bold] clusters with Claude "
            f"[dim]({args.model})[/dim]…"
        )
        console.print()

        try:
            labels = await label_clusters_concurrent(
                clusters_commits,
                keywords_per_cluster,
                on_label=_on_label,
                model=args.model,
                max_concurrent=args.concurrency,
            )
        except Exception as exc:
            console.print(f"[yellow]Warning: AI labeling failed: {exc}[/yellow]")
            console.print("[dim]Continuing with keyword-based labels.[/dim]")
            from .analysis import ThemeLabel
            labels = [
                ThemeLabel(
                    title=" / ".join(kws[:3]).title() if kws else "Uncategorised",
                    summary="",
                    action_items="None",
                )
                for kws in keywords_per_cluster
            ]

    # ── Phase 4: Print results ─────────────────────────────────────────────
    for i, (idx_list, label, kws) in enumerate(zip(clusters, labels, keywords_per_cluster), 1):
        cluster_commits = [commits[j] for j in idx_list]
        print_theme_result(i, label, cluster_commits, kws)

    # ── Phase 5: Overall summary ───────────────────────────────────────────
    if not args.no_ai and len(labels) >= 2:
        theme_titles = [lb.title for lb in labels]
        summary_parts: list[str] = []

        try:
            from .analysis import stream_overall_summary
            stream_overall_summary(
                commits,
                theme_titles,
                on_token=lambda chunk: summary_parts.append(chunk),
                model=args.model,
            )
        except Exception:
            pass

        if summary_parts:
            print_overall_summary("".join(summary_parts))

    # ── Phase 6: Export ───────────────────────────────────────────────────
    overall_text = ""
    if args.output:
        out_path = Path(args.output)
        fmt = args.export_format or (
            "markdown" if out_path.suffix in (".md", ".markdown") else "json"
        )
        from .export import to_json, to_markdown

        if fmt == "markdown":
            text = to_markdown(commits, clusters, labels, repo, overall_text)
        else:
            text = to_json(commits, clusters, labels, repo, overall_text)

        out_path.write_text(text, encoding="utf-8")
        console.print(f"  [dim]Exported {fmt} report to {args.output}[/dim]")
        console.print()

    # Summary footer
    clustered_count = sum(len(g) for g in clusters)
    unclustered = len(commits) - clustered_count
    console.print(
        f"  [bold green]Done.[/bold green]  "
        f"{len(clusters)} theme{'s' if len(clusters) != 1 else ''}  ·  "
        f"{clustered_count} clustered  ·  "
        f"[dim]{unclustered} unclustered[/dim]"
    )
    console.print()

    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _build_parser(cfg: dict) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gitsight",
        description="Semantic git history analysis — clusters commits into themes with AI labels.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  gitsight .                              # analyze current repo (last 200 commits)
  gitsight /path/to/repo --max-commits 500
  gitsight . --since "3 months ago"       # last 3 months
  gitsight . --author "Alice"             # filter by author
  gitsight . --no-ai                      # clustering only, no Claude API
  gitsight . --threshold 0.3              # tighter clusters
  gitsight . --output report.md           # export Markdown report
  gitsight . --output report.json         # export JSON report
  gitsight . --timeline                   # show weekly commit frequency chart
  gitsight . --author-stats               # show per-author velocity table

config file (~/.gitsight.toml or .gitsight.toml):
  model            = "claude-haiku-4-5-20251001"
  threshold        = 0.20
  min_cluster_size = 2
  concurrency      = 3
  max_commits      = 200
  no_ai            = false
  timeline         = false
  author_stats     = false

environment:
  ANTHROPIC_API_KEY   Required for AI labeling (set --no-ai to skip)
""",
    )
    p.add_argument("--version", action="version", version=f"gitsight {__version__}")
    p.add_argument(
        "repo",
        nargs="?",
        default=".",
        help="Path to the git repository (default: current directory)",
    )
    p.add_argument(
        "--max-commits", "-n",
        type=int,
        default=cfg.get("max_commits", 200),
        metavar="N",
        help="Maximum commits to analyze (default: 200)",
    )
    p.add_argument(
        "--since",
        metavar="DATE",
        help='Only commits after this date, e.g. "3 months ago", "2026-01-01"',
    )
    p.add_argument(
        "--until",
        metavar="DATE",
        help="Only commits before this date",
    )
    p.add_argument(
        "--author",
        metavar="PATTERN",
        help="Filter commits by author name or email",
    )
    p.add_argument(
        "--branch",
        default=cfg.get("branch", "HEAD"),
        metavar="REF",
        help="Branch or ref to analyze (default: HEAD)",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=cfg.get("threshold", 0.20),
        metavar="FLOAT",
        help="Cosine similarity threshold for clustering 0.0–1.0 (default: 0.20)",
    )
    p.add_argument(
        "--min-cluster-size",
        type=int,
        default=cfg.get("min_cluster_size", 2),
        metavar="N",
        help="Minimum commits for a cluster to be reported (default: 2)",
    )
    p.add_argument(
        "--no-ai",
        action="store_true",
        default=cfg.get("no_ai", False),
        help="Skip Claude labeling — show vector clusters with keyword labels only",
    )
    p.add_argument(
        "--model",
        default=cfg.get("model", "claude-haiku-4-5-20251001"),
        metavar="MODEL",
        help="Claude model ID (default: claude-haiku-4-5-20251001)",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=cfg.get("concurrency", 3),
        metavar="N",
        help="Max concurrent Claude API calls (default: 3)",
    )
    p.add_argument(
        "--output", "-o",
        metavar="FILE",
        help="Write a report to FILE (.md for Markdown, .json for JSON)",
    )
    p.add_argument(
        "--export-format",
        choices=["json", "markdown"],
        default=cfg.get("export_format", None),
        help="Force export format (inferred from --output extension by default)",
    )
    p.add_argument(
        "--timeline",
        action="store_true",
        default=cfg.get("timeline", False),
        help="Print a weekly commit-frequency bar chart before clustering",
    )
    p.add_argument(
        "--author-stats",
        action="store_true",
        default=cfg.get("author_stats", False),
        help="Print a per-author velocity table (commit count, cadence, date range)",
    )
    return p


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the analysis."""
    try:
        from .config import load_config
        cfg = load_config()
    except ValueError as exc:
        print(f"gitsight: config error: {exc}", file=sys.stderr)
        return 2

    parser = _build_parser(cfg)
    args = parser.parse_args(argv)

    try:
        return asyncio.run(_run_analysis(args))
    except KeyboardInterrupt:
        return 130


def entry_point() -> None:
    sys.exit(main())
