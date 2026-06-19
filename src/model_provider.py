from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProviderConfig:
    """Provider configuration shared by the agents.

    Supported providers for this lab:
    - openai
    - custom (OpenAI-compatible base URL)
    - gemini
    - anthropic
    - ollama
    - openrouter
    """

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


# Common typos / aliases mapped to the canonical provider name.
_PROVIDER_ALIASES = {
    "openai": "openai",
    "oai": "openai",
    "gpt": "openai",
    "chatgpt": "openai",
    "custom": "custom",
    "openai-compatible": "custom",
    "gemini": "gemini",
    "google": "gemini",
    "google-genai": "gemini",
    "anthropic": "anthropic",
    "anthorpic": "anthropic",  # frequent typo
    "claude": "anthropic",
    "ollama": "ollama",
    "local": "ollama",
    "openrouter": "openrouter",
    "open-router": "openrouter",
}

SUPPORTED_PROVIDERS = {
    "openai",
    "custom",
    "gemini",
    "anthropic",
    "ollama",
    "openrouter",
}


def normalize_provider(value: str) -> str:
    """Map aliases / typos to a canonical provider name.

    Example: `anthorpic` -> `anthropic`, `gpt` -> `openai`.
    """

    key = (value or "").strip().lower()
    if key in _PROVIDER_ALIASES:
        return _PROVIDER_ALIASES[key]
    if key in SUPPORTED_PROVIDERS:
        return key
    raise ValueError(
        f"Unsupported provider: {value!r}. "
        f"Supported: {', '.join(sorted(SUPPORTED_PROVIDERS))}."
    )


def build_chat_model(config: ProviderConfig):
    """Instantiate a LangChain chat model for the selected provider.

    SDKs are imported lazily inside each branch so that running in offline
    mode (or with only one provider installed) never fails because another
    provider's package is missing.
    """

    provider = normalize_provider(config.provider)
    temperature = config.temperature

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.model_name,
            temperature=temperature,
            api_key=config.api_key,
        )

    if provider == "custom":
        # OpenAI-compatible endpoint reached through ChatOpenAI + base_url.
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.model_name,
            temperature=temperature,
            api_key=config.api_key,
            base_url=config.base_url,
        )

    if provider == "openrouter":
        # OpenRouter is also OpenAI-compatible; default to its public base URL.
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.model_name,
            temperature=temperature,
            api_key=config.api_key,
            base_url=config.base_url or "https://openrouter.ai/api/v1",
        )

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=config.model_name,
            temperature=temperature,
            google_api_key=config.api_key,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=config.model_name,
            temperature=temperature,
            api_key=config.api_key,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=config.model_name,
            temperature=temperature,
            base_url=config.base_url or "http://localhost:11434",
        )

    raise ValueError(f"Unsupported provider: {config.provider!r}")
