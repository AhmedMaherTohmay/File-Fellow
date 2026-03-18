"""
Domain models for users, conversations, and conversation turns.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class User:
    """A registered user. Minimal anchor — expanded when auth is added."""
    id:         str
    created_at: datetime


@dataclass
class Conversation:
    """A single chat session belonging to a user."""
    id:         str
    user_id:    str
    created_at: datetime


@dataclass
class ConversationTurn:
    """
    A single message (user or assistant) within a conversation.

    The embedding allows semantic recall of past turns — when a user
    asks something, we search past turns for related context and inject
    it into the prompt alongside the retrieved document chunks.
    """
    id:              int
    user_id:         str
    conversation_id: str
    role:            str          # "user" or "assistant"
    content:         str
    created_at:      datetime
    embedding:       Optional[list[float]] = field(default=None, repr=False)
