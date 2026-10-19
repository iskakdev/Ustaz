"""Тьютор по материалам курса.

Три режима выбираются на лету:
- explain   - отвечает по найденным фрагментам лекций со ссылками [1], [2];
- socratic  - у студента открыт проверочный вопрос, и он спрашивает про него: готовый ответ не даём, ведём вопросами;
- off_topic - в материалах ничего не нашлось: честно говорим об этом, модель даже не вызываем.
"""
import math
import re
from operator import itemgetter
from langchain_core.documents import Document
from langchain_core.messages import trim_messages
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from ai import llm
from ai.history import SQLChatHistory
from ai.knowledge import CourseRetriever
from ai.loader import format_docs

# Косинус bge-m3 между вопросом тьютору и открытым проверочным вопросом: перефраз даёт ~0.8,
# соседние темы того же курса 0.3-0.5, близкие (лимит рекурсии к вопросу про RecursionError) ~0.68
SOCRATIC_SIM = 0.72
ASKS_ANSWER = re.compile(r'этот вопрос|текущ\w* вопрос|ответ на (этот |мой )?вопрос|как ответить|'
                         r'правильн\w* ответ|реши за меня|скажи ответ', re.IGNORECASE)

CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    ('system', 'Перепиши последний вопрос студента так, чтобы он был понятен без истории диалога. '
               'Сохрани язык вопроса. Верни только сам вопрос.'),
    MessagesPlaceholder('history'),
    ('human', '{question}'),
])

EXPLAIN_PROMPT = ChatPromptTemplate.from_messages([
    ('system', 'Ты тьютор курса «{course}». Отвечай только по фрагментам лекций ниже. После каждого утверждения '
               'ставь номер фрагмента, на который опираешься: [1], [2]. Если во фрагментах ответа нет, так и скажи. '
               'Пиши на языке вопроса, на "ты", '
               'коротко и по делу.\n\nФрагменты:\n{context}'),
    MessagesPlaceholder('history'),
    ('human', '{question}'),
])

SOCRATIC_PROMPT = ChatPromptTemplate.from_messages([
    ('system', 'Ты тьютор курса «{course}». Студент сейчас отвечает на проверочный вопрос: «{open_question}». '
               'Готовый ответ, решение или эталонную формулировку давать нельзя, даже если просят. '
               'Помоги думать: задай один наводящий вопрос или дай одну подсказку, опираясь на фрагменты лекций. '
               'Пиши на языке вопроса, на "ты", 1-3 предложения.\n\nФрагменты:\n{context}'),
    MessagesPlaceholder('history'),
    ('human', '{question}'),
])


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def trim(x: dict) -> list:
    return trim_messages(x['history'], max_tokens=1200, token_counter='approximate', start_on='human')


def find_docs(x: dict) -> list[Document]:
    return CourseRetriever(course_id=x['course_id']).invoke(x['standalone'])


async def afind_docs(x: dict) -> list[Document]:
    return await CourseRetriever(course_id=x['course_id']).ainvoke(x['standalone'])


def about_open_question(x: dict, similarity) -> bool:
    if not x.get('open_question'):
        return False
    # Явная просьба - по исходному тексту: переписанный с историей вопрос тянет в себя тему открытого вопроса
    if ASKS_ANSWER.search(x['question']):
        return True
    return similarity() >= SOCRATIC_SIM


def pick_mode(x: dict) -> str:
    def similarity():
        return cosine(*llm.get_embeddings().embed_documents([x['standalone'], x['open_question']]))

    if about_open_question(x, similarity):
        return 'socratic'
    return 'explain' if x['docs'] else 'off_topic'


async def apick_mode(x: dict) -> str:
    vectors = None
    if x.get('open_question'):
        vectors = await llm.get_embeddings().aembed_documents([x['standalone'], x['open_question']])
    if about_open_question(x, lambda: cosine(*vectors)):
        return 'socratic'
    return 'explain' if x['docs'] else 'off_topic'


def off_topic(x: dict) -> str:
    return (f'В материалах курса «{x["course"]}» об этом нет, поэтому выдумывать не буду. '
            f'Спроси про темы курса: {", ".join(x["topics"][:6])}.')


def sources(docs: list[Document]) -> list[dict]:
    return [{'n': n, 'section': d.metadata.get('section'), 'material': d.metadata.get('material'),
             'score': d.metadata.get('score')} for n, d in enumerate(docs, start=1)]


def build_tutor():
    model = llm.get_chat_model()
    condense = RunnableBranch(
        (lambda x: not x['history'], itemgetter('question')),
        CONDENSE_PROMPT | model | StrOutputParser(),
    )
    answer = RunnableBranch(
        (lambda x: x['mode'] == 'off_topic', RunnableLambda(off_topic)),
        (lambda x: x['mode'] == 'socratic', SOCRATIC_PROMPT | model | StrOutputParser()),
        EXPLAIN_PROMPT | model | StrOutputParser(),
    )
    chain = (
        RunnablePassthrough.assign(history=trim)
        | RunnablePassthrough.assign(standalone=condense)
        | RunnablePassthrough.assign(docs=RunnableLambda(find_docs, afunc=afind_docs))
        | RunnablePassthrough.assign(mode=RunnableLambda(pick_mode, afunc=apick_mode),
                                     context=lambda x: format_docs(x['docs']))
        | RunnablePassthrough.assign(answer=answer)
    )
    return RunnableWithMessageHistory(chain, lambda session_id: SQLChatHistory(session_id),
                                      input_messages_key='question', history_messages_key='history',
                                      output_messages_key='answer')


def session_id(student_id: int, course_id: int) -> str:
    return f'tutor:{student_id}:{course_id}'
