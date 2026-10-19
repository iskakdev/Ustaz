from typing import Sequence
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict
from database.db import SessionLocal
from database.models import ChatMessage


class SQLChatHistory(BaseChatMessageHistory):
    """История диалога в таблице chat_message. Её подключает RunnableWithMessageHistory."""

    def __init__(self, session_id: str, limit: int = 30):
        self.session_id = session_id
        self.limit = limit

    @property
    def messages(self) -> list[BaseMessage]:
        db = SessionLocal()
        try:
            rows = (db.query(ChatMessage).filter(ChatMessage.session_id == self.session_id)
                    .order_by(ChatMessage.id.desc()).limit(self.limit).all())
            return messages_from_dict([row.message for row in reversed(rows)])
        finally:
            db.close()

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        db = SessionLocal()
        try:
            db.add_all([ChatMessage(session_id=self.session_id, message=message_to_dict(m)) for m in messages])
            db.commit()
        finally:
            db.close()

    def clear(self) -> None:
        db = SessionLocal()
        try:
            db.query(ChatMessage).filter(ChatMessage.session_id == self.session_id).delete()
            db.commit()
        finally:
            db.close()
