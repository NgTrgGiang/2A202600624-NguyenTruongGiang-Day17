from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import (
    FACT_LABELS,
    CompactMemoryManager,
    UserProfileStore,
    estimate_tokens,
    extract_profile_updates,
)
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Agent B: short-term + persistent `User.md` + compact memory."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        # 1. Extract stable facts and 2. persist them into User.md.
        for key, value in extract_profile_updates(message).items():
            self.profile_store.upsert_fact(user_id, key, value)

        # 3. Append into compact memory (older content folds into a summary).
        self.compact_memory.append(thread_id, "user", message)

        # 4. Estimate the context this turn actually carries.
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = (
            self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        )

        # 5. Generate a response grounded in persisted memory.
        answer = self._offline_response(user_id, thread_id, message)

        # 6. Record the assistant turn and update token counters.
        self.compact_memory.append(thread_id, "assistant", answer)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + estimate_tokens(answer)

        return {
            "response": answer,
            "thread_id": thread_id,
            "prompt_tokens": prompt_tokens,
            "compactions": self.compaction_count(thread_id),
        }

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """Context carried into one turn = User.md + summary + recent messages.

        Crucially this is bounded by the compact threshold instead of growing
        with the whole thread, which is why long threads stay cheap.
        """

        profile = self.profile_store.read_text(user_id)
        ctx = self.compact_memory.context(thread_id)
        total = estimate_tokens(profile)
        total += estimate_tokens(str(ctx.get("summary", "")))
        for m in ctx.get("messages", []):  # type: ignore[union-attr]
            total += estimate_tokens(m.get("content", ""))
        return total

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        """Answer recall questions from persisted facts (works across threads)."""

        facts = self.profile_store.facts(user_id)
        lowered = message.lower()

        is_question = "?" in message or any(
            q in lowered
            for q in ("nhắc lại", "là gì", "là ai", "ở đâu", "nghề gì", "tóm tắt", "mô tả", "biết")
        )
        if not is_question:
            return "Mình đã ghi nhớ thông tin này vào hồ sơ của bạn."

        if not facts:
            return "Mình chưa có đủ thông tin về bạn để trả lời."

        # Pick which facts the question is asking about; default to a full
        # profile recap so any expected fact is covered.
        wanted: list[str] = []
        keyword_map = {
            "name": ("tên", "ai", "mô tả", "tóm tắt", "biết"),
            "location": ("ở đâu", "nơi ở", "ở"),
            "profession": ("nghề", "làm", "công việc"),
            "response_style": ("style", "trả lời", "kiểu"),
            "drink": ("uống", "đồ uống"),
            "food": ("món ăn", "ăn"),
            "pet": ("nuôi", "con gì", "thú cưng"),
            "interests": ("quan tâm", "thích", "kỹ thuật"),
        }
        for key, kws in keyword_map.items():
            if key in facts and any(kw in lowered for kw in kws):
                wanted.append(key)
        if not wanted:
            wanted = list(facts.keys())

        parts = [f"{FACT_LABELS.get(k, k)}: {facts[k]}" for k in wanted if k in facts]
        return "Theo hồ sơ mình nhớ về bạn — " + "; ".join(parts) + "."

    def _maybe_build_langchain_agent(self):
        """Optionally build a live agent; returns None in offline mode.

        A live design would combine `build_chat_model(self.config.model)`, an
        `InMemorySaver` for thread state, tools to read/write `User.md`, a
        dynamic prompt injecting profile memory, and summarization middleware.
        """

        if self.force_offline:
            return None
        return None  # offline-first lab: live wiring intentionally left inert
