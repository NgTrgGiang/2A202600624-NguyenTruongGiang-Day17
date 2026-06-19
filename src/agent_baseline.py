from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Agent A: within-session memory only.

    - Remembers turns inside one thread id.
    - Has no persistent `User.md`.
    - Forgets everything when a new thread starts (no cross-session recall).
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        # Offline deterministic path is the benchmark default. A live agent
        # would be wired through self.langchain_agent, but offline keeps the
        # comparison reproducible without API keys.
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        st = self.sessions.get(thread_id)
        return st.token_usage if st else 0

    def prompt_token_usage(self, thread_id: str) -> int:
        st = self.sessions.get(thread_id)
        return st.prompt_tokens_processed if st else 0

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        st = self.sessions.setdefault(thread_id, SessionState())
        st.messages.append({"role": "user", "content": message})

        # Baseline naively reprocesses the *entire* thread history every turn.
        # On long threads this cost grows quadratically -- the weakness compact
        # memory is meant to fix.
        history_tokens = sum(estimate_tokens(m["content"]) for m in st.messages)
        st.prompt_tokens_processed += history_tokens

        answer = self._offline_response(st, message)
        st.messages.append({"role": "assistant", "content": answer})
        st.token_usage += estimate_tokens(answer)

        return {
            "response": answer,
            "thread_id": thread_id,
            "prompt_tokens": history_tokens,
            "compactions": 0,
        }

    def _offline_response(self, st: SessionState, message: str) -> str:
        """Answer using only what was said in this thread.

        If a recall question arrives in a fresh thread, there is no history to
        draw from, so the baseline honestly admits it does not know.
        """

        lowered = message.lower()
        is_question = "?" in message or any(
            q in lowered for q in ("nhắc lại", "là gì", "là ai", "ở đâu", "nghề gì")
        )
        if is_question:
            # Search only this thread's user turns.
            found = [
                m["content"]
                for m in st.messages
                if m["role"] == "user" and m["content"] != message
            ]
            if not found:
                return "Mình chưa có thông tin nào trong phiên này để trả lời."
            return "Trong phiên này bạn có nhắc: " + " ".join(found[-3:])
        return "Đã ghi nhận trong phiên hiện tại."

    def _maybe_build_langchain_agent(self):
        """Optionally build a live agent; returns None in offline mode.

        A real implementation would use `build_chat_model(self.config.model)`
        with an `InMemorySaver` so short-term memory lives per thread.
        """

        if self.force_offline:
            return None
        return None  # offline-first lab: live wiring intentionally left inert
