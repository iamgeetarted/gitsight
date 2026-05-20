"""Command-line interface for gitsight."""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
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

    t_start = time.perf_counter()
    repo = Path(args.repo).resolve()
    if not (repo / ".git").exists() and not (repo / "HEAD").exists():
        print(f"gitsight: not a git repository: {repo}", file=sys.stderr)
        return 2

    # ── Phase 1: read commits ──────────────────────────────────────────────
    t0 = time.perf_counter()
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
    t_git = time.perf_counter() - t0

    if not commits:
        print("gitsight: no commits found matching the given filters.")
        return 0

    if getattr(args, "verbose", False):
        console.print(f"  [dim]git log: {len(commits)} commits in {t_git:.2f}s[/dim]")

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
    t0 = time.perf_counter()
    messages = [c.message for c in commits]
    matrix, vocab = embed_messages(messages)
    clusters = greedy_cluster(matrix, threshold=args.threshold, min_size=args.min_cluster_size)
    t_vec = time.perf_counter() - t0

    if getattr(args, "verbose", False):
        console.print(f"  [dim]vector embed+cluster: vocab={len(vocab)} terms, {len(clusters)} clusters, {t_vec:.3f}s[/dim]")

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

        t0 = time.perf_counter()
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
        t_ai = time.perf_counter() - t0
        if getattr(args, "verbose", False):
            console.print(f"  [dim]AI labeling: {len(labels)} labels in {t_ai:.2f}s[/dim]")

    # ── Phase 4: Print results ─────────────────────────────────────────────
    for i, (idx_list, label, kws) in enumerate(zip(clusters, labels, keywords_per_cluster), 1):
        cluster_commits = [commits[j] for j in idx_list]
        print_theme_result(i, label, cluster_commits, kws)

    # ── Phase 5: Overall summary ───────────────────────────────────────────
    overall_text = ""
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
            overall_text = "".join(summary_parts)
            print_overall_summary(overall_text)

    # ── Phase 6: Export ───────────────────────────────────────────────────
    if args.output:
        out_path = Path(args.output)
        fmt = args.export_format or (
            "csv" if out_path.suffix == ".csv" else
            "markdown" if out_path.suffix in (".md", ".markdown") else
            "json"
        )
        from .export import to_json, to_markdown, to_csv

        if fmt == "markdown":
            text = to_markdown(commits, clusters, labels, repo, overall_text)
        elif fmt == "csv":
            text = to_csv(commits, clusters, labels)
        else:
            text = to_json(commits, clusters, labels, repo, overall_text)

        out_path.write_text(text, encoding="utf-8")
        console.print(f"  [dim]Exported {fmt} report to {args.output}[/dim]")
        console.print()

    # Summary footer
    clustered_count = sum(len(g) for g in clusters)
    unclustered = len(commits) - clustered_count
    t_total = time.perf_counter() - t_start
    timing_str = f"  [dim]({t_total:.1f}s)[/dim]" if getattr(args, "verbose", False) else ""
    console.print(
        f"  [bold green]Done.[/bold green]  "
        f"{len(clusters)} theme{'s' if len(clusters) != 1 else ''}  ·  "
        f"{clustered_count} clustered  ·  "
        f"[dim]{unclustered} unclustered[/dim]"
        + timing_str
    )
    console.print()

    return 0


# ---------------------------------------------------------------------------
# Hotfiles subcommand
# ---------------------------------------------------------------------------

async def _run_hotfiles(args: argparse.Namespace) -> int:
    """Run the file churn heatmap subcommand."""
    from .hotfiles import read_file_churn_async, render_hotfiles
    from rich.console import Console

    console = Console()
    repo = Path(args.repo).resolve()
    if not (repo / ".git").exists() and not (repo / "HEAD").exists():
        print(f"gitsight: not a git repository: {repo}", file=sys.stderr)
        return 2

    console.print(f"  [dim]Reading file churn for [bold]{repo.name}[/bold]…[/dim]")
    try:
        entries = await read_file_churn_async(
            repo,
            max_count=args.max_commits,
            since=getattr(args, "since", None),
            until=getattr(args, "until", None),
            author=getattr(args, "author", None),
            branch=getattr(args, "branch", "HEAD"),
            min_commits=args.min_commits,
        )
    except ValueError as exc:
        print(f"gitsight: {exc}", file=sys.stderr)
        return 2

    render_hotfiles(entries, top_n=args.top)

    if args.output:
        import csv as _csv, io
        buf = io.StringIO()
        writer = _csv.writer(buf, lineterminator="\n")
        writer.writerow(["rank", "path", "commit_count", "lines_added", "lines_deleted", "churn_score"])
        for i, e in enumerate(entries[:args.top], 1):
            writer.writerow([i, e.path, e.commit_count, e.lines_added, e.lines_deleted, e.churn_score])
        Path(args.output).write_text(buf.getvalue(), encoding="utf-8")
        console.print(f"  [dim]Saved CSV to {args.output}[/dim]")

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
  gitsight . --output report.csv          # export CSV (one row per commit)
  gitsight . --timeline                   # show weekly commit frequency chart
  gitsight . --author-stats               # show per-author velocity table
  gitsight . --verbose                    # show timing for each phase
  gitsight hotfiles .                     # file churn heatmap
  gitsight hotfiles . --top 30            # show top 30 hottest files
  gitsight hotfiles . --output churn.csv  # save churn data to CSV

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

    sub = p.add_subparsers(dest="command", metavar="COMMAND")

    # ── default (analyse) subcommand ──────────────────────────────────────
    def _add_common_filters(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("repo", nargs="?", default=".", help="Path to git repository (default: .)")
        sp.add_argument("--max-commits", "-n", type=int, default=cfg.get("max_commits", 200), metavar="N")
        sp.add_argument("--since", metavar="DATE", help='E.g. "3 months ago"')
        sp.add_argument("--until", metavar="DATE")
        sp.add_argument("--author", metavar="PATTERN")
        sp.add_argument("--branch", default=cfg.get("branch", "HEAD"), metavar="REF")

    # analyse (default)
    p_analyse = sub.add_parser("analyse", aliases=["analyze"], help="Cluster commits into themes (default)")
    _add_common_filters(p_analyse)
    p_analyse.add_argument("--threshold", type=float, default=cfg.get("threshold", 0.20), metavar="FLOAT")
    p_analyse.add_argument("--min-cluster-size", type=int, default=cfg.get("min_cluster_size", 2), metavar="N")
    p_analyse.add_argument("--no-ai", action="store_true", default=cfg.get("no_ai", False))
    p_analyse.add_argument("--model", default=cfg.get("model", "claude-haiku-4-5-20251001"), metavar="MODEL")
    p_analyse.add_argument("--concurrency", type=int, default=cfg.get("concurrency", 3), metavar="N")
    p_analyse.add_argument("--output", "-o", metavar="FILE")
    p_analyse.add_argument("--export-format", choices=["json", "markdown", "csv"], default=cfg.get("export_format", None))
    p_analyse.add_argument("--timeline", action="store_true", default=cfg.get("timeline", False))
    p_analyse.add_argument("--author-stats", action="store_true", default=cfg.get("author_stats", False))
    p_analyse.add_argument("--verbose", "-v", action="store_true", help="Show timing for each pipeline phase")
    p_analyse.set_defaults(func=lambda a: asyncio.run(_run_analysis(a)))

    # hotfiles
    p_hot = sub.add_parser("hotfiles", help="Show file churn heatmap — which files change most")
    _add_common_filters(p_hot)
    p_hot.add_argument("--top", type=int, default=20, metavar="N", help="Show top N files (default: 20)")
    p_hot.add_argument("--min-commits", type=int, default=2, metavar="N", help="Min commits to include a file (default: 2)")
    p_hot.add_argument("--output", "-o", metavar="FILE", help="Save churn data as CSV")
    p_hot.set_defaults(func=lambda a: asyncio.run(_run_hotfiles(a)))

    # Legacy top-level flags for backwards compatibility (no subcommand)
    p.add_argument("repo_positional", nargs="?", default=None, help=argparse.SUPPRESS)
    p.add_argument("--max-commits", "-n", type=int, default=cfg.get("max_commits", 200), metavar="N")
    p.add_argument("--since", metavar="DATE")
    p.add_argument("--until", metavar="DATE")
    p.add_argument("--author", metavar="PATTERN")
    p.add_argument("--branch", default=cfg.get("branch", "HEAD"), metavar="REF")
    p.add_argument("--threshold", type=float, default=cfg.get("threshold", 0.20), metavar="FLOAT")
    p.add_argument("--min-cluster-size", type=int, default=cfg.get("min_cluster_size", 2), metavar="N")
    p.add_argument("--no-ai", action="store_true", default=cfg.get("no_ai", False))
    p.add_argument("--model", default=cfg.get("model", "claude-haiku-4-5-20251001"), metavar="MODEL")
    p.add_argument("--concurrency", type=int, default=cfg.get("concurrency", 3), metavar="N")
    p.add_argument("--output", "-o", metavar="FILE")
    p.add_argument("--export-format", choices=["json", "markdown", "csv"], default=cfg.get("export_format", None))
    p.add_argument("--timeline", action="store_true", default=cfg.get("timeline", False))
    p.add_argument("--author-stats", action="store_true", default=cfg.get("author_stats", False))
    p.add_argument("--verbose", "-v", action="store_true")

    return p


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the selected subcommand."""
    try:
        from .config import load_config
        cfg = load_config()
    except ValueError as exc:
        print(f"gitsight: config error: {exc}", file=sys.stderr)
        return 2

    parser = _build_parser(cfg)
    args = parser.parse_args(argv)

    # If a known subcommand was selected, delegate to it
    if hasattr(args, "func"):
        try:
            return args.func(args) or 0
        except KeyboardInterrupt:
            return 130

    # Fallback: run the legacy top-level analysis (no subcommand given)
    # Map repo_positional → repo for the analysis function
    if args.repo_positional:
        args.repo = args.repo_positional
    else:
        args.repo = "."

    try:
        return asyncio.run(_run_analysis(args))
    except KeyboardInterrupt:
        return 130


def entry_point() -> None:
    sys.exit(main())
