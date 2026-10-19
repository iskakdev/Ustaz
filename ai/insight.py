"""Бриф для преподавателя: в чём группа путается и что разобрать на следующем занятии.

Map: по каждому трудному понятию модель читает неверные ответы и называет общий паттерн.
Reduce: из паттернов собирается план на занятие с приоритетами.
"""
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from ai import llm


class Difficulty(BaseModel):
    pattern: str = Field(description='В чём именно путаются студенты, одной-двумя фразами')
    example: str = Field(description='Короткая цитата из неверного ответа, которая это показывает')


class Priority(BaseModel):
    concept: str
    why: str = Field(description='Почему это важно разобрать, со ссылкой на паттерн ошибок')
    how: str = Field(description='Как объяснить иначе: пример, аналогия или упражнение')


class TeacherBrief(BaseModel):
    summary: str = Field(description='Общая картина в 2-3 предложениях')
    priorities: list[Priority] = Field(description='До трёх тем для следующего занятия, самая важная первой')


MAP_PROMPT = ChatPromptTemplate.from_messages([
    ('system', 'Ты помогаешь преподавателю курса «{course}». Ниже неверные и неполные ответы студентов '
               'по одному понятию. Найди общий паттерн ошибки, а не пересказывай каждый ответ. Пиши по-русски.'),
    ('human', 'Понятие: {concept}\nСуть: {summary}\nЧастые заблуждения: {misconceptions}\n\nОтветы:\n{answers}'),
])

REDUCE_PROMPT = ChatPromptTemplate.from_messages([
    ('system', 'Ты методист курса «{course}». По паттернам ошибок составь план следующего занятия: '
               'до трёх тем, самая важная первой. Учитывай карту: если путаются в базовом понятии, '
               'его надо разобрать раньше зависимых. Пиши по-русски, конкретно.'),
    ('human', 'Паттерны ошибок:\n{patterns}\n\nКарта (понятие <- предпосылки):\n{map}'),
])


async def build_brief(course: str, struggles: list[dict], course_map: str) -> TeacherBrief:
    model = llm.get_chat_model()
    patterns = await (MAP_PROMPT | model.with_structured_output(Difficulty)).abatch(
        [{'course': course, **s} for s in struggles],
        config={**llm.run_config('insight.map'), 'max_concurrency': 2})
    lines = '\n'.join(f'- {s["concept"]} (средний балл {s["avg_score"]:.0%}): {p.pattern} Пример: «{p.example}»'
                      for s, p in zip(struggles, patterns))
    reduce_chain = REDUCE_PROMPT | model.with_structured_output(TeacherBrief)
    return await reduce_chain.ainvoke({'course': course, 'patterns': lines, 'map': course_map},
                                      config=llm.run_config('insight.reduce'))
