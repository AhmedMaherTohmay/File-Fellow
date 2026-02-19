"""
LLM factory.
Returns a cached LangChain chat-model instance.
Currently supports Groq only.
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
    """
    Return a cached LangChain ChatGroq instance.
    """

    if LLM_PROVIDER.lower() != "groq":
        raise ValueError(
            f"Unsupported LLM_PROVIDER='{LLM_PROVIDER}'. "
            "Only 'groq' is currently supported."
        )

    if not LLM_KEY:
        raise RuntimeError("LLM_KEY (LLM API Key) is not set in environment variables.")

    try:
        from langchain_groq import ChatGroq

        llm = ChatGroq(
            model=GROQ_MODEL_ID,
            api_key=LLM_KEY,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )

        logger.info(
            "Initialized Groq LLM | model=%s | temperature=%s | max_tokens=%s",
            GROQ_MODEL_ID,
            LLM_TEMPERATURE,
            LLM_MAX_TOKENS,
        )

        return llm

    except ImportError:
        logger.error("langchain-groq package not found. Install it with 'pip install langchain-groq'.")
        raise
    except Exception as e:
        logger.exception("Failed to initialize Groq LLM")
        raise RuntimeError(f"Groq LLM initialization failed: {e}") from e