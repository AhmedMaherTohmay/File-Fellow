"""
LLM factory.
Returns a cached LangChain chat-model instance (Groq).
"""
from __future__ import annotations

import logging
from functools import lru_cache

from config.settings import settings
from src.core.exceptions import LLMError

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_llm():
    """
    Return a cached LangChain ChatGroq instance.

    Raises:
        LLMError: If the provider is unsupported, the API key is missing,
                  or ChatGroq initialization fails.
    """
    if settings.LLM_PROVIDER.lower() != "groq":
        raise LLMError(
            f"Unsupported LLM_PROVIDER='{settings.LLM_PROVIDER}'. "
            "Only 'groq' is currently supported. "
            "Set LLM_PROVIDER=groq in your .env file."
        )

    if not settings.LLM_KEY:
        raise LLMError(
            "LLM_KEY is not set. Please add LLM_KEY=<your-groq-api-key> to your .env file."
        )

    try:
        from langchain_groq import ChatGroq

        llm = ChatGroq(
            model=settings.GROQ_MODEL_ID,
            api_key=settings.LLM_KEY,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
        )

        logger.info(
            "Initialized Groq LLM | model=%s | temperature=%.1f | max_tokens=%d",
            settings.GROQ_MODEL_ID,
            settings.LLM_TEMPERATURE,
            settings.LLM_MAX_TOKENS,
        )
        return llm

    except LLMError:
        raise  # already typed, let it propagate
    except Exception as exc:
        logger.exception("Failed to initialize Groq LLM")
        raise LLMError(f"Groq LLM initialization failed: {exc}") from exc


@lru_cache(maxsize=1)
def get_llm_for_eval():
    """
    Return an LLM instance for evaluation (judge role).

    Uses a slightly higher temperature for answer diversity.

    Raises:
        LLMError: If initialization fails.
    """
    if not settings.LLM_KEY:
        raise LLMError(
            "LLM_KEY is not set. Cannot initialize eval LLM."
        )

    try:
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=settings.GROQ_MODEL_ID,
            api_key=settings.LLM_KEY,
            temperature=0.1,
            max_tokens=512,
        )
    except Exception as exc:
        logger.error("Failed to create eval LLM: %s", exc)
        raise LLMError(f"Eval LLM initialization failed: {exc}") from exc
