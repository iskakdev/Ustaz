import time
from uuid import UUID
from typing import Any
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from database.db import SessionLocal
from database.models import AIUsage


class UsageCallback(BaseCallbackHandler):
    """Пишет каждый вызов модели в ai_usage: фича, модель, токены, задержка, был ли кэш."""
    run_inline = True

    def __init__(self, feature: str):
        self.feature = feature
        self.started: dict[UUID, tuple[float, str]] = {}

    def on_chat_model_start(self, serialized: dict, messages: list, *, run_id: UUID, **kwargs: Any):
        model = (kwargs.get('metadata') or {}).get('ls_model_name') or serialized.get('name', '?')
        self.started[run_id] = (time.perf_counter(), model)

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any):
        usage = {}
        if response.generations and response.generations[0]:
            message = getattr(response.generations[0][0], 'message', None)
            usage = getattr(message, 'usage_metadata', None) or {}
        cached = usage.get('total_cost') == 0  # BaseChatModel обнуляет стоимость на попаданиях в кэш
        self.save(run_id, ok=True, cached=cached,
                  input_tokens=0 if cached else usage.get('input_tokens', 0),
                  output_tokens=0 if cached else usage.get('output_tokens', 0))

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any):
        self.save(run_id, ok=False, cached=False, input_tokens=0, output_tokens=0)

    def save(self, run_id: UUID, **fields):
        started, model = self.started.pop(run_id, (time.perf_counter(), '?'))
        db = SessionLocal()
        try:
            db.add(AIUsage(feature=self.feature, model=model,
                           latency_ms=int((time.perf_counter() - started) * 1000), **fields))
            db.commit()
        finally:
            db.close()
