from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config
from memory_store import UserProfileStore, extract_profile_updates


def make_config(tmp_path: Path):
    """Isolated config: state in tmp_path, low compact threshold for fast tests."""

    config = load_config()
    config.state_dir = tmp_path / "state"
    config.state_dir.mkdir(parents=True, exist_ok=True)
    config.compact_threshold_tokens = 60
    config.compact_keep_messages = 2
    return config


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")

    # Default profile before anything is written.
    assert store.read_text("u1").startswith("# User Profile")
    assert store.file_size("u1") == 0

    store.write_text("u1", "# User Profile\n- name: An\n")
    assert "name: An" in store.read_text("u1")
    assert store.file_size("u1") > 0

    # edit_text replaces an existing fragment.
    assert store.edit_text("u1", "An", "Bình") is True
    assert "name: Bình" in store.read_text("u1")
    assert store.edit_text("u1", "khong-ton-tai", "x") is False

    # upsert_fact overwrites the same key instead of duplicating it.
    store.upsert_fact("u1", "name", "Cường")
    store.upsert_fact("u1", "city", "Huế")
    facts = store.facts("u1")
    assert facts["name"] == "Cường"
    assert facts["city"] == "Huế"
    assert store.read_text("u1").count("- name:") == 1


def test_compact_trigger(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    agent = AdvancedAgent(config, force_offline=True)

    long_turn = "Đây là một câu rất dài để ép compact memory phải nén lịch sử cũ lại. " * 2
    for _ in range(8):
        agent.reply("user", "thread-long", long_turn)

    assert agent.compaction_count("thread-long") > 0


def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    advanced = AdvancedAgent(config, force_offline=True)
    baseline = BaselineAgent(config, force_offline=True)

    intro = "Chào bạn, mình tên là DũngCT."
    for agent in (advanced, baseline):
        agent.reply("dungct", "session-1", intro)

    question = "Mình tên là gì?"
    adv_answer = advanced.reply("dungct", "session-2", question)["response"]
    base_answer = baseline.reply("dungct", "session-2", question)["response"]

    # Advanced remembers across a brand new thread; baseline does not.
    assert "DũngCT" in adv_answer
    assert "DũngCT" not in base_answer


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    advanced = AdvancedAgent(config, force_offline=True)
    baseline = BaselineAgent(config, force_offline=True)

    long_turn = "Mình kể thêm rất nhiều chi tiết dài để thread phình to dần theo thời gian. " * 2
    for i in range(12):
        advanced.reply("user", "t-adv", f"{long_turn} (lượt {i})")
        baseline.reply("user", "t-base", f"{long_turn} (lượt {i})")

    adv_prompt = advanced.prompt_token_usage("t-adv")
    base_prompt = baseline.prompt_token_usage("t-base")

    # Compaction keeps the advanced context bounded, so it carries far less.
    assert advanced.compaction_count("t-adv") > 0
    assert adv_prompt < base_prompt


def test_conflict_handling_keeps_only_latest_fact(tmp_path: Path) -> None:
    """Bonus: a correction replaces the stale value instead of keeping both."""

    config = make_config(tmp_path)
    agent = AdvancedAgent(config, force_offline=True)

    agent.reply("u", "s1", "Mình đang làm backend engineer cho một startup.")
    agent.reply("u", "s1", "Mình không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer.")

    profile_text = agent.profile_store.read_text("u")
    facts = agent.profile_store.facts("u")

    assert facts["profession"] == "MLOps engineer"
    assert "backend engineer" not in profile_text


def test_extract_skips_questions(tmp_path: Path) -> None:
    """Bonus guardrail: questions must not be stored as facts."""

    assert extract_profile_updates("Mình tên là gì?") == {}
    assert extract_profile_updates("Nhắc lại giúp mình tên và nghề nghiệp.") == {}
    assert extract_profile_updates("Mình tên là DũngCT.")["name"] == "DũngCT"
