from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def recall_points(answer: str, expected: list[str]) -> float:
    """Return 1.0 if all expected facts appear, 0.5 if some, 0.0 if none."""

    if not expected:
        return 0.0
    low = (answer or "").lower()
    hits = sum(1 for e in expected if e.lower() in low)
    if hits == len(expected):
        return 1.0
    if hits > 0:
        return 0.5
    return 0.0


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Lightweight offline quality: coverage + sensible structure/length."""

    if not answer:
        return 0.0
    coverage = 0.0
    if expected:
        low = answer.lower()
        coverage = sum(1 for e in expected if e.lower() in low) / len(expected)
    structure = 1.0 if 10 <= len(answer) <= 400 else 0.5
    return round(0.7 * coverage + 0.3 * structure, 3)


def run_agent_benchmark(
    agent_name: str, agent, conversations: list[dict[str, Any]], config
) -> BenchmarkRow:
    agent_tokens = 0
    prompt_tokens = 0
    compactions = 0
    recall_scores: list[float] = []
    quality_scores: list[float] = []
    seen_users: set[str] = set()

    # Phase 1: feed every conversation (one thread each) so all sessions and
    # corrections settle before we measure cross-session recall.
    for conv in conversations:
        user_id = conv["user_id"]
        thread_id = conv["id"]
        seen_users.add(user_id)
        for turn in conv["turns"]:
            agent.reply(user_id, thread_id, turn)
        agent_tokens += agent.token_usage(thread_id)
        prompt_tokens += agent.prompt_token_usage(thread_id)
        compactions += agent.compaction_count(thread_id)

    # Phase 2: ask recall questions in FRESH threads (true cross-session test).
    for conv in conversations:
        user_id = conv["user_id"]
        for i, q in enumerate(conv.get("recall_questions", [])):
            recall_thread = f"{conv['id']}-recall-{i}"
            result = agent.reply(user_id, recall_thread, q["question"])
            answer = result["response"]
            expected = q.get("expected_contains", [])
            recall_scores.append(recall_points(answer, expected))
            quality_scores.append(heuristic_quality(answer, expected))
            agent_tokens += agent.token_usage(recall_thread)
            prompt_tokens += agent.prompt_token_usage(recall_thread)

    # Memory growth: advanced persists User.md; baseline has no file.
    memory_bytes = 0
    if hasattr(agent, "memory_file_size"):
        memory_bytes = sum(agent.memory_file_size(u) for u in seen_users)

    avg_recall = round(sum(recall_scores) / len(recall_scores), 3) if recall_scores else 0.0
    avg_quality = round(sum(quality_scores) / len(quality_scores), 3) if quality_scores else 0.0

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=agent_tokens,
        prompt_tokens_processed=prompt_tokens,
        recall_score=avg_recall,
        response_quality=avg_quality,
        memory_growth_bytes=memory_bytes,
        compactions=compactions,
    )


_HEADERS = [
    "Agent",
    "Agent tokens only",
    "Prompt tokens processed",
    "Cross-session recall",
    "Response quality",
    "Memory growth (bytes)",
    "Compactions",
]


def format_rows(rows: list[BenchmarkRow]) -> str:
    table = [
        [
            r.agent_name,
            r.agent_tokens_only,
            r.prompt_tokens_processed,
            r.recall_score,
            r.response_quality,
            r.memory_growth_bytes,
            r.compactions,
        ]
        for r in rows
    ]
    try:
        from tabulate import tabulate

        return tabulate(table, headers=_HEADERS, tablefmt="github")
    except ImportError:
        lines = [" | ".join(_HEADERS), " | ".join(["---"] * len(_HEADERS))]
        for row in table:
            lines.append(" | ".join(str(c) for c in row))
        return "\n".join(lines)


def _run_suite(title: str, dataset: Path, config) -> None:
    conversations = load_conversations(dataset)
    rows = [
        run_agent_benchmark("Baseline", BaselineAgent(config, force_offline=True), conversations, config),
        run_agent_benchmark("Advanced", AdvancedAgent(config, force_offline=True), conversations, config),
    ]
    print(f"\n## {title}\n")
    print(format_rows(rows))


def main() -> None:
    config = load_config(Path(__file__).resolve().parent.parent)

    _run_suite("Standard Benchmark", config.data_dir / "conversations.json", config)
    _run_suite(
        "Long-Context Stress Benchmark",
        config.data_dir / "advanced_long_context.json",
        config,
    )


if __name__ == "__main__":
    main()
