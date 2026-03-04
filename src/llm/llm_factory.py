"""
LLM factory.
Returns a cached LangChain chat-model instance (Groq).
"""
from __future__ import annotations

import logging
from functools import lru_cache

from config.settings import (
    LLM_PROVIDER,
    LLM_KEY,
    GROQ_MODEL_ID,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
)

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_llm():
    """Return a cached LangChain ChatGroq instance.

    Raises:
        ValueError: If the provider is not supported.
        RuntimeError: If the API key is missing or initialization fails.
    """
    if LLM_PROVIDER.lower() != "groq":
        raise ValueError(
            f"Unsupported LLM_PROVIDER='{LLM_PROVIDER}'. "
            "Only 'groq' is currently supported. "
            "Set LLM_PROVIDER=groq in your .env file."
        )

    if not LLM_KEY:
        raise RuntimeError(
            "LLM_KEY is not set. Please add LLM_KEY=<your-groq-api-key> to your .env file."
        )

    try:
        from langchain_groq import ChatGroq

        llm = ChatGroq(
            model=GROQ_MODEL_ID,
            api_key=LLM_KEY,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )

        logger.info(
            "Initialized Groq LLM | model=%s | temperature=%.1f | max_tokens=%d",
            GROQ_MODEL_ID,
            LLM_TEMPERATURE,
            LLM_MAX_TOKENS,
        )
        return llm

    except Exception as e:
        logger.exception("Failed to initialize Groq LLM")
        raise RuntimeError(f"Groq LLM initialization failed: {e}") from e

@lru_cache(maxsize=1)
def get_llm_for_eval():
    """Return an LLM instance suitable for evaluation (judge role).
    Uses the same model but with slightly higher temperature for diversity.
    """
    try:
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=GROQ_MODEL_ID,
            api_key=LLM_KEY,
            temperature=0.1,
            max_tokens=512,
        )
    except Exception as e:
        logger.error("Failed to create eval LLM: %s", e)
        raise
