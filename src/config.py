from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig, normalize_provider


@dataclass
class LabConfig:
    """Shared configuration for the lab.

    - Paths for the repo root, dataset directory, and state directory.
    - Compact-memory settings (threshold and number of messages to keep).
    - Provider settings for the main model and the judge model.
    """

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


# Which environment variable holds the API key for each provider.
_API_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "custom": "CUSTOM_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "ollama": None,  # local, no key
    "openrouter": "OPENROUTER_API_KEY",
}

# Sensible default model name per provider when LLM_MODEL is not set.
_DEFAULT_MODEL = {
    "openai": "gpt-4o-mini",
    "custom": "gpt-4o-mini",
    "gemini": "gemini-1.5-flash",
    "anthropic": "claude-haiku-4-5-20251001",
    "ollama": "llama3.1",
    "openrouter": "openai/gpt-4o-mini",
}


def _load_dotenv(root: Path) -> None:
    """Load `.env` if python-dotenv is available; silently skip otherwise."""

    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_path = root / ".env"
    if env_path.exists():
        load_dotenv(env_path)


def _build_provider_config(provider: str, model_name: str, temperature: float) -> ProviderConfig:
    provider = normalize_provider(provider)
    key_env = _API_KEY_ENV.get(provider)
    api_key = os.getenv(key_env) if key_env else None

    base_url = None
    if provider == "custom":
        base_url = os.getenv("CUSTOM_BASE_URL")
    elif provider == "openrouter":
        base_url = os.getenv("OPENROUTER_BASE_URL")
    elif provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL")

    return ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
    )


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load environment variables and return a populated LabConfig.

    1. Resolve the repo root (or default to this file's parent's parent).
    2. Load values from `.env` when possible.
    3. Create `state/` if it does not exist.
    4. Build provider configs for the main and judge models.
    """

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    _load_dotenv(root)

    data_dir = root / "data"
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    provider = normalize_provider(os.getenv("LLM_PROVIDER", "openai"))
    model_name = os.getenv("LLM_MODEL", _DEFAULT_MODEL[provider])
    temperature = float(os.getenv("LLM_TEMPERATURE", "0"))

    model = _build_provider_config(provider, model_name, temperature)

    # Judge model defaults to the same provider/model but can be overridden.
    judge_provider = normalize_provider(os.getenv("JUDGE_PROVIDER", provider))
    judge_model_name = os.getenv("JUDGE_MODEL", _DEFAULT_MODEL[judge_provider])
    judge_model = _build_provider_config(judge_provider, judge_model_name, 0.0)

    # Compact memory defaults: kept small so compaction is easy to observe.
    compact_threshold_tokens = int(os.getenv("COMPACT_THRESHOLD_TOKENS", "400"))
    compact_keep_messages = int(os.getenv("COMPACT_KEEP_MESSAGES", "4"))

    return LabConfig(
        base_dir=root,
        data_dir=data_dir,
        state_dir=state_dir,
        compact_threshold_tokens=compact_threshold_tokens,
        compact_keep_messages=compact_keep_messages,
        model=model,
        judge_model=judge_model,
    )
