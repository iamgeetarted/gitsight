"""Export analysis results to JSON, Markdown, and CSV."""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .git import Commit
    from .analysis import ThemeLabel


def to_json(
    commits: list[Commit],
    clusters: list[list[int]],
    labels: list[ThemeLabel],
    repo_path: Path,
    overall_summary: str = "",
) -> str:
    """Serialise analysis results to a JSON string.

    Args:
        commits: All commits analyzed.
        clusters: Index-lists grouping commit positions.
        labels: ThemeLabel for each cluster.
        repo_path: Repository root path.
        overall_summary: Optional narrative summary text.

    Returns:
        Indented JSON string.
    """
    themes = []
    for cluster_indices, label in zip(clusters, labels):
        cluster_commits = [commits[i] for i in cluster_indices]
        themes.append({
            "title": label.title,
            "summary": label.summary,
            "action_items": label.action_items,
            "commit_count": len(cluster_commits),
            "date_range": {
                "earliest": min(c.date_str for c in cluster_commits),
                "latest": max(c.date_str for c in cluster_commits),
            },
            "authors": list({c.author for c in cluster_commits}),
            "commits": [
                {"sha": c.sha[:8], "date": c.date_str, "author": c.author, "subject": c.subject}
                for c in cluster_commits
            ],
        })

    return json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo": str(repo_path.resolve()),
        "total_commits": len(commits),
        "themes_found": len(themes),
        "overall_summary": overall_summary,
        "themes": themes,
    }, indent=2)


def to_markdown(
    commits: list[Commit],
    clusters: list[list[int]],
    labels: list[ThemeLabel],
    repo_path: Path,
    overall_summary: str = "",
) -> str:
    """Render analysis results as a Markdown document.

    Args:
        commits: All commits analyzed.
        clusters: Index-lists grouping commit positions.
        labels: ThemeLabel for each cluster.
        repo_path: Repository root path.
        overall_summary: Optional narrative summary text.

    Returns:
        Markdown string.
    """
    lines: list[str] = [
        f"# gitsight Report — `{repo_path.name}`",
        "",
        f"*Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*  ",
        f"*Analysed {len(commits)} commits across {len(labels)} themes*",
        "",
    ]

    if overall_summary:
        lines += ["## Executive Summary", "", overall_summary, ""]

    lines += ["## Themes", ""]

    for i, (cluster_indices, label) in enumerate(zip(clusters, labels), 1):
        cluster_commits = [commits[j] for j in cluster_indices]
        earliest = min(c.date_str for c in cluster_commits)
        latest = max(c.date_str for c in cluster_commits)
        authors = sorted({c.author for c in cluster_commits})

        lines += [
            f"### {i}. {label.title}",
            "",
            f"**{len(cluster_commits)} commits** · {earliest} → {latest} · "
            + ", ".join(authors[:3]) + (f" +{len(authors)-3}" if len(authors) > 3 else ""),
            "",
            label.summary,
            "",
        ]
        if label.action_items and label.action_items.lower() != "none":
            lines += [f"> **Action:** {label.action_items}", ""]

        lines += ["<details><summary>Commits</summary>", ""]
        lines += ["| SHA | Date | Author | Subject |", "|-----|------|--------|---------|"]
        for c in cluster_commits[:20]:
            subj = c.subject.replace("|", "\\|")
            lines.append(f"| `{c.sha[:8]}` | {c.date_str} | {c.author} | {subj} |")
        if len(cluster_commits) > 20:
            lines.append(f"| … | … | … | *+{len(cluster_commits)-20} more* |")
        lines += ["", "</details>", ""]

    # Unclustered commits
    clustered = {i for group in clusters for i in group}
    unclustered = [c for i, c in enumerate(commits) if i not in clustered]
    if unclustered:
        lines += [
            "## Unclustered Commits",
            "",
            f"*{len(unclustered)} commits did not fit any theme.*",
            "",
        ]

    return "\n".join(lines)


def to_csv(
    commits: list[Commit],
    clusters: list[list[int]],
    labels: list[ThemeLabel],
) -> str:
    """Serialise analysis results to a CSV string.

    Each row represents one commit with its assigned theme title.
    Commits not assigned to any cluster are labelled "Unclustered".

    Args:
        commits: All commits analyzed.
        clusters: Index-lists grouping commit positions.
        labels: ThemeLabel for each cluster.

    Returns:
        CSV string with header row.
    """
    index_to_theme: dict[int, str] = {}
    for cluster_indices, label in zip(clusters, labels):
        for i in cluster_indices:
            index_to_theme[i] = label.title

    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["sha", "date", "author", "theme", "subject"])
    for i, c in enumerate(commits):
        theme = index_to_theme.get(i, "Unclustered")
        writer.writerow([c.sha[:8], c.date_str, c.author, theme, c.subject])
    return buf.getvalue()
