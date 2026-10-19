import hashlib
from typing import Any, Optional, Sequence
from langchain_core.caches import BaseCache
from langchain_core.messages import message_to_dict, messages_from_dict
from langchain_core.outputs import ChatGeneration, Generation
from database.db import SessionLocal
from database.models import LLMCache


class SQLCache(BaseCache):
    """Кэш ответов модели в той же SQLite. Ключ - промпт плюс параметры модели (температура, формат).
    Пересборка курса после правки одного файла не гоняет модель по неизменённым разделам,
    а один и тот же ответ на один и тот же вопрос всегда получает одну и ту же оценку."""

    @staticmethod
    def make_key(prompt: str, llm_string: str) -> str:
        return hashlib.sha256(f'{llm_string}\n{prompt}'.encode()).hexdigest()

    def lookup(self, prompt: str, llm_string: str) -> Optional[Sequence[Generation]]:
        db = SessionLocal()
        try:
            row = db.query(LLMCache).filter(LLMCache.key == self.make_key(prompt, llm_string)).first()
            if not row:
                return None
            try:
                generations = [ChatGeneration(message=messages_from_dict([item['message']])[0],
                                              generation_info=item.get('generation_info')) for item in row.generations]
            except (KeyError, TypeError, ValueError):
                return None  # запись старого формата - просто спросим модель заново
            row.hits += 1
            db.commit()
            return generations
        finally:
            db.close()

    def update(self, prompt: str, llm_string: str, return_val: Sequence[Generation]) -> None:
        db = SessionLocal()
        try:
            key = self.make_key(prompt, llm_string)
            if not db.query(LLMCache).filter(LLMCache.key == key).first():
                db.add(LLMCache(key=key, generations=[
                    {'message': message_to_dict(item.message), 'generation_info': item.generation_info}
                    for item in return_val if isinstance(item, ChatGeneration)]))
                db.commit()
        finally:
            db.close()

    def clear(self, **kwargs: Any) -> None:
        db = SessionLocal()
        try:
            db.query(LLMCache).delete()
            db.commit()
        finally:
            db.close()
