from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    """Cheap, stable token estimator for offline benchmarking.

    Not tied to a real tokenizer: ~4 characters per token is good enough to
    compare two agents consistently.
    """

    if not text:
        return 0
    stripped = text.strip()
    if not stripped:
        return 0
    return max(1, len(stripped) // 4)


# --------------------------------------------------------------------------- #
# Persistent user profile (User.md)
# --------------------------------------------------------------------------- #

_DEFAULT_PROFILE = "# User Profile\n"
_FACT_LINE = re.compile(r"^-\s*([\w_]+)\s*:\s*(.+?)\s*$")

# Human-friendly labels used when rendering a profile back to the user.
FACT_LABELS = {
    "name": "Tên",
    "location": "Nơi ở hiện tại",
    "profession": "Nghề nghiệp hiện tại",
    "response_style": "Style trả lời mong muốn",
    "drink": "Đồ uống yêu thích",
    "food": "Món ăn yêu thích",
    "pet": "Thú cưng",
    "interests": "Mối quan tâm kỹ thuật",
}


def _slugify(user_id: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", user_id.strip().lower())
    return slug.strip("_") or "user"


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`, one markdown file per user id."""

    root_dir: Path

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, user_id: str) -> Path:
        return self.root_dir / f"{_slugify(user_id)}.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if not path.exists():
            return _DEFAULT_PROFILE
        return path.read_text(encoding="utf-8")

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        """Replace one occurrence inside User.md. Returns True if it changed."""

        content = self.read_text(user_id)
        if search_text not in content:
            return False
        self.write_text(user_id, content.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    # ---- structured fact helpers (bonus: entity extraction) ---------------- #

    def facts(self, user_id: str) -> dict[str, str]:
        """Parse `- key: value` lines from User.md into an ordered dict."""

        out: dict[str, str] = {}
        for line in self.read_text(user_id).splitlines():
            m = _FACT_LINE.match(line)
            if m:
                out[m.group(1)] = m.group(2).strip()
        return out

    def upsert_fact(self, user_id: str, key: str, value: str) -> bool:
        """Insert or overwrite a single fact. Returns True if it changed.

        Overwriting is how conflict handling works: a new correction replaces
        the stale value instead of keeping both.
        """

        value = value.strip()
        if not value:
            return False

        content = self.read_text(user_id)
        if not content.strip():
            content = _DEFAULT_PROFILE

        lines = content.splitlines()
        new_line = f"- {key}: {value}"
        for i, line in enumerate(lines):
            m = _FACT_LINE.match(line)
            if m and m.group(1) == key:
                if line.strip() == new_line:
                    return False  # unchanged
                lines[i] = new_line
                self.write_text(user_id, "\n".join(lines) + "\n")
                return True

        # Key not present yet: append it.
        if lines and lines[-1].strip() == "":
            lines = lines[:-1]
        lines.append(new_line)
        self.write_text(user_id, "\n".join(lines) + "\n")
        return True


# --------------------------------------------------------------------------- #
# Fact extraction (with question / noise / conflict guardrails)
# --------------------------------------------------------------------------- #

# Gazetteer keeps "where the user lives" apart from places merely mentioned.
_KNOWN_LOCATIONS = [
    "Đà Nẵng",
    "Huế",
    "Hà Nội",
    "Hồ Chí Minh",
    "Sài Gòn",
    "TP HCM",
    "Hải Phòng",
    "Cần Thơ",
    "Nha Trang",
    "Đà Lạt",
]
# Prepositions that signal residence (NOT "từ"=from / "nhắc"=mention).
_RESIDENCE_PREP = ("ở", "tại", "sang", "về", "đến", "chuyển tới", "chuyển đến")

_NOISE_MARKERS = ("đùa", "bay ra họp", "vừa bay", "chỉ là nơi", "lúc đầu mình nói")

# Interrogative fillers that must never be stored as a real value, e.g.
# "đồ uống yêu thích của mình là gì." (a recall prompt phrased as a statement).
_QUESTION_VALUES = {"gì", "nào", "sao", "đâu", "ai", "bao nhiêu", "thế nào"}


def _clean_value(value: str) -> str | None:
    value = value.strip(" .,:")
    if not value:
        return None
    if value.lower() in _QUESTION_VALUES:
        return None
    return value

_NAME_RE = re.compile(
    r"(?:mình|tôi|tớ)\s+tên\s+(?:là\s+|gọi\s+là\s+)?([\wÀ-ỹĐđ]+(?:\s+[\wÀ-ỹĐđ]+){0,2})",
    re.IGNORECASE,
)
_JOB_RE = re.compile(r"([\wÀ-ỹĐđ]+\s+(?:engineer|manager|developer|scientist|lead))", re.IGNORECASE)
_DRINK_RE = re.compile(r"đồ uống[^.\n]*?(?:là|:)\s*([\wÀ-ỹĐđ\s]+?)(?:[.\n,]|$)", re.IGNORECASE)
_FOOD_RE = re.compile(r"món ăn[^.\n]*?(?:là|:)\s*([\wÀ-ỹĐđ\s]+?)(?:[.\n,]|$)", re.IGNORECASE)
_PET_RE = re.compile(
    r"nuôi\s+(?:một\s+|con\s+|bé\s+)*([\wÀ-ỹĐđ]+(?:\s+tên\s+[\wÀ-ỹĐđ]+)?)", re.IGNORECASE
)


_NEGATION_PATTERNS = (
    re.compile(r"chứ\s+không[^.,;]*", re.IGNORECASE),        # "...chứ không còn ở Đà Nẵng nữa"
    re.compile(r"không\s+còn\b[^.,;]*?\bnữa", re.IGNORECASE),  # "không còn làm backend engineer nữa"
    re.compile(r"không\s+phải[^.,;]*", re.IGNORECASE),         # "không phải là ..."
)


def _strip_negations(sentence: str) -> str:
    """Remove negated spans so only the *current* (corrected) fact remains.

    Handles both shapes the dataset uses:
    - "X chứ không còn Y nữa"          -> drops "chứ không còn Y nữa", keeps X
    - "không còn Y nữa, giờ chuyển X"  -> drops "không còn Y nữa", keeps X
    """

    cleaned = sentence
    for pat in _NEGATION_PATTERNS:
        cleaned = pat.sub(" ", cleaned)
    return cleaned


def _extract_location(text: str) -> str | None:
    best = None
    for city in _KNOWN_LOCATIONS:
        for m in re.finditer(re.escape(city), text):
            prefix = text[max(0, m.start() - 12): m.start()].lower()
            if any(prep in prefix for prep in _RESIDENCE_PREP):
                best = city  # last residence mention wins within the clause
    return best


def extract_profile_updates(message: str) -> dict[str, str]:
    """Convert raw user text into stable profile facts.

    Guardrails:
    - skip pure questions (avoid storing facts from "what's my name?")
    - skip noisy / joking clauses ("product manager"... "chỉ là câu đùa")
    - resolve corrections so the new value replaces the old one
    """

    updates: dict[str, str] = {}
    if not message or "?" in message:
        return updates
    # "Nhắc lại giúp mình ..." asks the agent to retrieve facts (a question in
    # disguise); but "Nhắc lại lần cuối: tên mình là ..." is the user restating
    # facts, which we DO want to learn.
    if "nhắc lại giúp" in message.lower():
        return updates

    sentences = re.split(r"(?<=[.!\n])\s+", message)
    for raw in sentences:
        sentence = raw.strip()
        if not sentence:
            continue
        lowered = sentence.lower()
        if any(noise in lowered for noise in _NOISE_MARKERS):
            continue

        clause = _strip_negations(sentence)

        # name (must be a proper noun: starts with an uppercase letter)
        m = _NAME_RE.search(clause)
        if m:
            cand = m.group(1).strip()
            if cand[:1].isupper():
                updates["name"] = cand

        # location (gazetteer + residence preposition)
        loc = _extract_location(clause)
        if loc:
            updates["location"] = loc

        # profession
        if any(k in clause.lower() for k in ("làm", "nghề", "chuyển sang", "engineer", "manager")):
            jm = _JOB_RE.search(clause)
            if jm:
                updates["profession"] = jm.group(1).strip()

        # response style (must mention a concrete style keyword)
        if ("trả lời" in lowered or "câu trả lời" in lowered) and (
            "ngắn gọn" in lowered or "bullet" in lowered
        ):
            start = lowered.find("ngắn gọn")
            if start == -1:
                start = lowered.find("bullet")
            style = sentence[start:].rstrip(" .")
            updates["response_style"] = style

        # drink / food
        dm = _DRINK_RE.search(clause)
        if dm and (val := _clean_value(dm.group(1))):
            updates["drink"] = val
        fm = _FOOD_RE.search(clause)
        if fm and (val := _clean_value(fm.group(1))):
            updates["food"] = val

        # pet
        pm = _PET_RE.search(clause)
        if pm and (val := _clean_value(pm.group(1))):
            updates["pet"] = val

        # technical interests
        has_python = re.search(r"\bpython\b", lowered) is not None
        has_ai = re.search(r"\bai\b", lowered) is not None
        if has_python or has_ai:
            parts = []
            if has_python:
                parts.append("Python")
            if has_ai:
                parts.append("AI ứng dụng")
            updates["interests"] = ", ".join(parts)

    return updates


# --------------------------------------------------------------------------- #
# Compact memory for long threads
# --------------------------------------------------------------------------- #

def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Heuristic compact summary of older messages.

    Keeps it short by truncating each message; this is what makes compaction
    actually reduce the carried context instead of just relocating it.
    """

    if not messages:
        return ""
    picked = messages[-max_items:] if len(messages) > max_items else messages
    bullets = []
    for m in picked:
        content = " ".join(m.get("content", "").split())
        if len(content) > 80:
            content = content[:77] + "..."
        bullets.append(f"{m.get('role', 'user')}: {content}")
    return "Tóm tắt hội thoại cũ: " + " | ".join(bullets)


@dataclass
class CompactMemoryManager:
    """Keep recent messages in full and fold older ones into a summary."""

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def _ensure(self, thread_id: str) -> dict[str, object]:
        if thread_id not in self.state:
            self.state[thread_id] = {"messages": [], "summary": "", "compactions": 0}
        return self.state[thread_id]

    def _total_tokens(self, st: dict[str, object]) -> int:
        total = estimate_tokens(str(st["summary"]))
        for m in st["messages"]:  # type: ignore[index]
            total += estimate_tokens(m.get("content", ""))
        return total

    def append(self, thread_id: str, role: str, content: str) -> None:
        st = self._ensure(thread_id)
        st["messages"].append({"role": role, "content": content})  # type: ignore[union-attr]

        # Compact while we exceed the budget and have more than we must keep.
        while (
            self._total_tokens(st) > self.threshold_tokens
            and len(st["messages"]) > self.keep_messages  # type: ignore[arg-type]
        ):
            messages = st["messages"]  # type: ignore[assignment]
            to_compact = messages[: -self.keep_messages]
            kept = messages[-self.keep_messages:]
            new_summary = summarize_messages(to_compact)
            st["summary"] = (str(st["summary"]) + " " + new_summary).strip()
            st["messages"] = kept
            st["compactions"] = int(st["compactions"]) + 1  # type: ignore[arg-type]

    def context(self, thread_id: str) -> dict[str, object]:
        return self._ensure(thread_id)

    def compaction_count(self, thread_id: str) -> int:
        return int(self._ensure(thread_id)["compactions"])  # type: ignore[arg-type]
