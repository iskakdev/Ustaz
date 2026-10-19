"""Коуч - агент, который видит прогресс студента через инструменты и помнит разговор (checkpointer).

Кто спрашивает, агент не угадывает: student_id и course_id приходят в runtime context из токена,
поэтому подсмотреть чужой прогресс нельзя, даже если попросить.
"""
from dataclasses import dataclass
from datetime import timedelta
from langchain.agents import create_agent
from langchain.agents.middleware import (ModelCallLimitMiddleware, ModelRequest, ModelResponse,
                                         SummarizationMiddleware, ToolCallLimitMiddleware, dynamic_prompt,
                                         wrap_model_call)
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from ai import llm
from ai.forge import find_same
from ai.knowledge import CourseRetriever
from database.db import SessionLocal, utcnow
from database.models import Attempt, Question, Concept, Verdict
from mastery import bkt, graph
from mastery.planner import kind_for
from mastery.state import student_states, status

MINUTES_PER_QUESTION = 3


@dataclass
class CoachContext:
    student_id: int
    course_id: int
    student_name: str
    course_title: str


@tool
def progress_overview(runtime: ToolRuntime[CoachContext]) -> str:
    """Прогресс студента по каждому понятию курса: процент знания, статус и когда повторять."""
    db = SessionLocal()
    try:
        now = utcnow()
        states, requires, _ = student_states(db, runtime.context.student_id, runtime.context.course_id)
        lines = []
        for s in sorted(states.values(), key=lambda s: (s.depth, s.id)):
            line = f'- {s.name}: {s.p:.0%}, {status(s, requires, states, now)}'
            if s.last_seen:
                line += f', повторить {bkt.next_review(s.last_seen, s.stability):%d.%m %H:%M} UTC'
            lines.append(line)
        mastered = sum(1 for s in states.values() if s.p >= bkt.MASTERED)
        return f'Освоено {mastered} из {len(states)} понятий.\n' + '\n'.join(lines)
    finally:
        db.close()


@tool
def why_stuck(concept: str, runtime: ToolRuntime[CoachContext]) -> str:
    """Почему понятие не даётся: цепочка слабых предпосылок до настоящей причины."""
    db = SessionLocal()
    try:
        states, requires, _ = student_states(db, runtime.context.student_id, runtime.context.course_id)
        name = find_same(concept, [s.name for s in states.values()])
        if not name:
            return f'Понятия «{concept}» в курсе нет. Есть: {", ".join(s.name for s in states.values())}.'
        target = next(s for s in states.values() if s.name == name)
        chain = graph.weakest_chain(target.id, requires, {cid: s.p for cid, s in states.items()}, bkt.UNLOCK)
        path = ' -> '.join(f'{states[cid].name} ({states[cid].p:.0%})' for cid in chain)
        if len(chain) == 1:
            return f'{path}. Предпосылки освоены, дело в самом понятии - нужна практика.'
        return f'{path}. Начинать стоит с «{states[chain[-1]].name}».'
    finally:
        db.close()


@tool
def plan_session(minutes: int, runtime: ToolRuntime[CoachContext]) -> str:
    """План занятия на заданное число минут: сначала то, что выцветает, потом новые понятия."""
    db = SessionLocal()
    try:
        now = utcnow()
        states, requires, _ = student_states(db, runtime.context.student_id, runtime.context.course_id)
        slots = max(1, minutes // MINUTES_PER_QUESTION)
        due = sorted((s for s in states.values() if status(s, requires, states, now) == 'fading'),
                     key=lambda s: bkt.next_review(s.last_seen, s.stability))
        fresh = sorted((s for s in states.values() if status(s, requires, states, now) in ('open', 'learning')),
                       key=lambda s: (-len(graph.descendants(s.id, requires)), s.depth))
        plan = [f'повторить «{s.name}» ({kind_for(s.p)})' for s in due][:slots]
        for s in fresh:
            if len(plan) >= slots:
                break
            need = 1 if s.p >= bkt.UNLOCK else 2
            plan += [f'разобрать «{s.name}» ({kind_for(s.p)})'] * min(need, slots - len(plan))
        if not plan:
            return 'Сейчас нечего делать: всё освоено и повторять рано.'
        return f'{len(plan)} вопросов по ~{MINUTES_PER_QUESTION} мин:\n' + '\n'.join(f'{n}. {p}' for n, p in
                                                                                  enumerate(plan, start=1))
    finally:
        db.close()


@tool
def recent_mistakes(runtime: ToolRuntime[CoachContext], limit: int = 5) -> str:
    """Последние ошибки студента: вопрос, оценка, заблуждение и разбор от проверяющего."""
    db = SessionLocal()
    try:
        rows = (db.query(Attempt).join(Question).join(Concept)
                .filter(Attempt.student_id == runtime.context.student_id,
                        Concept.course_id == runtime.context.course_id, Attempt.verdict != Verdict.correct,
                        Attempt.created_at >= utcnow() - timedelta(days=30))
                .order_by(Attempt.id.desc()).limit(limit).all())
        if not rows:
            return 'Ошибок за последний месяц нет.'
        return '\n'.join(f'- «{a.attempt_question.question_concept.name}»: {a.score:.0%}. '
                         f'{("Заблуждение: " + a.misconception + ". ") if a.misconception else ""}{a.feedback}'
                         for a in rows)
    finally:
        db.close()


@tool
async def search_course(query: str, runtime: ToolRuntime[CoachContext]) -> str:
    """Поиск по материалам курса, когда нужно объяснить или процитировать лекцию."""
    docs = await CourseRetriever(course_id=runtime.context.course_id, k=3).ainvoke(query)
    if not docs:
        return 'В материалах курса ничего не нашлось.'
    return '\n\n'.join(f'({d.metadata.get("section")}) {d.page_content}' for d in docs)


SYSTEM = ('Ты коуч на учебной платформе Ustaz. Помогаешь студенту понять, что учить и почему он застрял. '
          'Факты о прогрессе бери только из инструментов, проценты не выдумывай: как дела - progress_overview, '
          'почему не получается понятие - why_stuck, план занятия - plan_session, разбор ошибок - recent_mistakes, '
          'объяснить тему - search_course. Названия инструментов студенту не называй. '
          'Отвечай на языке студента, на "ты", коротко и по делу; в конце - один конкретный следующий шаг.')
GROUND = 'Ты ответил без данных. Сначала вызови подходящий инструмент, потом отвечай.'


@dynamic_prompt
def coach_prompt(request: ModelRequest) -> str:
    ctx: CoachContext = request.runtime.context
    return f'{SYSTEM}\nСтудент: {ctx.student_name}. Курс: «{ctx.course_title}». Сейчас {utcnow():%d.%m.%Y %H:%M} UTC.'


@wrap_model_call
async def ground_first(request: ModelRequest, handler) -> ModelResponse:
    """7B-модель охотно отвечает «из головы», и инструменты зовёт только на первом сообщении.
    Если на новое сообщение студента она не вызвала ни одного инструмента, переспрашиваем один раз с требованием.
    tool_choice='any' тут не поможет: Ollama не умеет принуждать к вызову инструмента."""
    response = await handler(request)
    asked = bool(request.messages) and isinstance(request.messages[-1], HumanMessage)
    answer = response.result[-1] if response.result else None
    if asked and isinstance(answer, AIMessage) and not answer.tool_calls:
        system = request.system_message.text if request.system_message else ''
        return await handler(request.override(system_message=SystemMessage(f'{system}\n{GROUND}')))
    return response


TOOLS = [progress_overview, why_stuck, plan_session, recent_mistakes, search_course]
_agent = None


def get_coach():
    """Агент один на процесс: в его InMemorySaver живут диалоги всех студентов, каждый под своим thread_id."""
    global _agent
    if _agent is None:
        model = llm.get_chat_model()
        _agent = create_agent(
            model, tools=TOOLS, context_schema=CoachContext, checkpointer=InMemorySaver(),
            middleware=[
                coach_prompt,
                ground_first,
                ModelCallLimitMiddleware(run_limit=6, exit_behavior='end'),
                ToolCallLimitMiddleware(run_limit=6),
                SummarizationMiddleware(model, trigger=('tokens', 3000), keep=('messages', 10)),
            ])
    return _agent


def reset_coach():
    global _agent
    _agent = None


async def ask_coach(context: CoachContext, message: str) -> dict:
    config = {**llm.run_config('coach', student_id=context.student_id),
              'configurable': {'thread_id': f'coach:{context.student_id}:{context.course_id}'}}
    result = await get_coach().ainvoke({'messages': [HumanMessage(message)]}, config=config, context=context)
    turn = []
    for m in reversed(result['messages']):
        if isinstance(m, HumanMessage):
            break
        turn.append(m)
    tools_used = [m.name for m in reversed(turn) if isinstance(m, ToolMessage)]
    return {'answer': result['messages'][-1].text, 'tools_used': tools_used}
