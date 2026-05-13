"""AI-powered commit analysis — streams theme labels from Claude."""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass
from typing import AsyncIterator, Callable

from .git import Commit


@dataclass
class ThemeLabel:
    """Claude's semantic label for a cluster of commits."""

    title: str          # Short theme title, e.g. "Authentication refactoring"
    summary: str        # 1-2 sentence description
    action_items: str   # Optional recommended follow-up actions


def _get_client():
    """Return an Anthropic client or raise with a helpful message."""
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "pip install anthropic  # required for AI analysis in gitsight"
        )
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY is not set.\n"
            "Export it:  export ANTHROPIC_API_KEY=sk-ant-..."
        )
    import anthropic
    return anthropic.Anthropic(api_key=api_key)


def _cluster_prompt(commits: list[Commit], keywords: list[str]) -> str:
    lines = "\n".join(
        f"{i+1}. [{c.date_str}] {c.subject}"
        for i, c in enumerate(commits[:30])
    )
    omitted = max(0, len(commits) - 30)
    kw_str = ", ".join(keywords) if keywords else "(none detected)"

    return (
        f"You are analyzing a cluster of related git commits from a software project.\n\n"
        f"Detected keywords: {kw_str}\n\n"
        f"Commits ({len(commits)} total{f', showing 30' if omitted else ''}):\n"
        f"{lines}\n"
        + (f"(+ {omitted} more)\n" if omitted else "")
        + "\n"
        "Respond with exactly 3 lines (no headers, no blank lines):\n"
        "LINE 1: A short theme title (3-6 words, title case)\n"
        "LINE 2: A 1-2 sentence description of what these commits are doing and why\n"
        "LINE 3: One concrete action item an engineer should consider (or 'None')\n\n"
        "Be specific and technical. Use the actual language of the code when possible."
    )


def _overall_prompt(commits: list[Commit], themes: list[str]) -> str:
    theme_list = "\n".join(f"- {t}" for t in themes)
    recent = "\n".join(
        f"[{c.date_str}] {c.subject}" for c in commits[:20]
    )
    return (
        "You are writing an executive summary of a git repository's recent activity.\n\n"
        f"Identified themes:\n{theme_list}\n\n"
        f"Most recent commits:\n{recent}\n\n"
        "Write a 3-5 sentence narrative summary for a senior engineer or tech lead:\n"
        "• What is the overall direction of recent work?\n"
        "• Any concerning patterns (e.g. many hotfixes, large churning areas)?\n"
        "• What appears to be the highest-velocity area of development?\n\n"
        "Be direct and specific. Skip pleasantries."
    )


async def label_cluster_streaming(
    commits: list[Commit],
    keywords: list[str],
    on_token: Callable[[str], None],
    model: str = "claude-haiku-4-5-20251001",
) -> ThemeLabel:
    """Stream a Claude label for one commit cluster, calling *on_token* per chunk.

    Args:
        commits: Commits in this cluster.
        keywords: Top TF-IDF keywords detected for this cluster.
        on_token: Callback invoked with each streamed text chunk.
        model: Claude model ID.

    Returns:
        Parsed ThemeLabel after streaming completes.
    """
    client = _get_client()
    prompt = _cluster_prompt(commits, keywords)

    full_text = ""
    with client.messages.stream(
        model=model,
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for chunk in stream.text_stream:
            on_token(chunk)
            full_text += chunk

    lines = [l.strip() for l in full_text.strip().splitlines() if l.strip()]
    title = lines[0] if len(lines) > 0 else "Miscellaneous changes"
    summary = lines[1] if len(lines) > 1 else ""
    actions = lines[2] if len(lines) > 2 else "None"

    return ThemeLabel(title=title, summary=summary, action_items=actions)


async def label_clusters_concurrent(
    clusters: list[list[Commit]],
    keywords_per_cluster: list[list[str]],
    on_label: Callable[[int, str], None] | None = None,
    model: str = "claude-haiku-4-5-20251001",
    max_concurrent: int = 3,
) -> list[ThemeLabel]:
    """Label all clusters concurrently using asyncio.TaskGroup.

    Uses a semaphore to cap concurrency at *max_concurrent* simultaneous
    Claude API calls — avoids rate-limit errors on large repos.

    Args:
        clusters: List of commit groups to label.
        keywords_per_cluster: Parallel list of keyword lists.
        on_label: Optional callback(cluster_index, full_text) called when
                  each label stream completes.
        model: Claude model ID.
        max_concurrent: Max simultaneous Claude calls.

    Returns:
        List of ThemeLabel objects, one per cluster.
    """
    sem = asyncio.Semaphore(max_concurrent)
    labels: list[ThemeLabel | None] = [None] * len(clusters)

    async def _label_one(idx: int, commits: list[Commit], kws: list[str]) -> None:
        buf: list[str] = []

        def _collect(chunk: str) -> None:
            buf.append(chunk)

        async with sem:
            label = await asyncio.to_thread(
                _label_one_sync, commits, kws, _collect, model
            )
        labels[idx] = label
        if on_label:
            on_label(idx, "".join(buf))

    def _label_one_sync(
        commits: list[Commit],
        kws: list[str],
        on_token: Callable[[str], None],
        model: str,
    ) -> ThemeLabel:
        client = _get_client()
        prompt = _cluster_prompt(commits, kws)
        full_text = ""
        with client.messages.stream(
            model=model,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for chunk in stream.text_stream:
                on_token(chunk)
                full_text += chunk
        lines = [l.strip() for l in full_text.strip().splitlines() if l.strip()]
        title = lines[0] if lines else "Miscellaneous"
        summary = lines[1] if len(lines) > 1 else ""
        actions = lines[2] if len(lines) > 2 else "None"
        return ThemeLabel(title=title, summary=summary, action_items=actions)

    async with asyncio.TaskGroup() as tg:
        for idx, (commits, kws) in enumerate(zip(clusters, keywords_per_cluster)):
            tg.create_task(_label_one(idx, commits, kws))

    return [lb for lb in labels if lb is not None]


def stream_overall_summary(
    commits: list[Commit],
    themes: list[str],
    on_token: Callable[[str], None],
    model: str = "claude-haiku-4-5-20251001",
) -> None:
    """Stream an overall narrative summary of all commits and themes to *on_token*.

    Args:
        commits: All commits analyzed.
        themes: List of theme titles already identified.
        on_token: Called with each text chunk as it streams.
        model: Claude model ID.
    """
    client = _get_client()
    prompt = _overall_prompt(commits, themes)

    with client.messages.stream(
        model=model,
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for chunk in stream.text_stream:
            on_token(chunk)
