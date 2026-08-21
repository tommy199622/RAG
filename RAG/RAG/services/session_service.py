"""
Conversation Session Service

負責：
1. 保存目前對話型號
2. 保存對話歷史
3. 取得 / 更新 Session
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from threading import Lock


@dataclass
class ConversationMessage:
    role: str
    content: str


@dataclass
class ConversationSession:
    current_model: Optional[str] = None

    history: List[ConversationMessage] = field(
        default_factory=list
    )


_sessions: Dict[str, ConversationSession] = {}

_lock = Lock()


def get_session(session_id: str) -> ConversationSession:

    with _lock:

        if session_id not in _sessions:

            _sessions[session_id] = (
                ConversationSession()
            )

        return _sessions[session_id]


def set_model(
    session_id: str,
    model: str
):

    session = get_session(session_id)

    session.current_model = model


def get_model(
    session_id: str
) -> Optional[str]:

    session = get_session(session_id)

    return session.current_model


def add_message(
    session_id: str,
    role: str,
    content: str
):

    session = get_session(session_id)

    session.history.append(
        ConversationMessage(
            role=role,
            content=content
        )
    )


def get_history(
    session_id: str,
    max_messages: int = 10
) -> List[ConversationMessage]:

    session = get_session(session_id)

    return session.history[-max_messages:]


def clear_session(
    session_id: str
):

    with _lock:

        _sessions.pop(
            session_id,
            None
        )