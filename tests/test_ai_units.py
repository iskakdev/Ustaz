import asyncio
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from conftest import ScriptedChatModel


def test_sql_cache_returns_same_answer_and_marks_usage(fakes):
    import main  # noqa: F401  включает SQLCache
    from ai.llm import run_config
    from database.db import SessionLocal
    from database.models import AIUsage, LLMCache

    model = GenericFakeChatModel(messages=iter([AIMessage('первый'), AIMessage('второй')]))
    first = model.invoke('Сколько будет 2+2? cache-test', config=run_config('cache-test'))
    second = model.invoke('Сколько будет 2+2? cache-test', config=run_config('cache-test'))
    assert first.text == second.text == 'первый'

    db = SessionLocal()
    assert max(row.hits for row in db.query(LLMCache).all()) >= 1
    usage = db.query(AIUsage).filter(AIUsage.feature == 'cache-test').order_by(AIUsage.id).all()
    assert [u.cached for u in usage] == [False, True]
    db.close()


def test_grader_falls_back_to_keywords_when_model_breaks(fakes, monkeypatch):
    from ai import grader, llm
    broken = ScriptedChatModel(script=lambda messages, kwargs: AIMessage('не знаю, что ответить'))
    monkeypatch.setattr(llm, 'get_chat_model', lambda *args, **kwargs: broken)
    grade, by = asyncio.run(grader.grade(
        'Курс', 'Стек вызовов', 'Что такое стек вызовов?', 'Эталон', ['кадр функции', 'принцип LIFO'], [],
        'Каждый вызов кладёт кадр функции на вершину, снимается по LIFO. fallback-test'))
    assert by == 'keywords' and grade.score == 1.0 and grade.missing == []


def test_misconception_must_be_in_the_answer():
    from ai.grader import Grade, grounded
    grade = lambda: Grade(score=0.2, covered=[], missing=['всё'], misconception='Мемоизация уменьшает глубину рекурсии',
                          feedback='-')
    assert grounded(grade(), 'Там что-то со стеком, точно не помню').misconception is None
    assert grounded(grade(), 'С мемоизацией глубина рекурсии уменьшается').misconception is not None
