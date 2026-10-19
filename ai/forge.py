"""Сборка курса: материалы -> разделы -> понятия с предпосылками -> карта -> вопросы трёх уровней."""
import difflib
import re
from operator import itemgetter
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel
from ai import llm
from ai.knowledge import CourseRetriever, build_store
from ai.loader import MaterialLoader, split_sections, split_chunks, format_docs
from database.db import SessionLocal
from database.models import Course, CourseStatus, Concept, ConceptLink, Question, QuestionKind
from mastery import graph
from mastery.planner import plural


class ExtractedConcept(BaseModel):
    name: str = Field(description='Название понятия, 1-4 слова, как в лекции')
    summary: str = Field(description='Суть понятия в 1-2 предложениях своими словами')
    # Без default: в JSON-схеме поле со значением по умолчанию необязательное, и модель его просто пропускает
    requires: list[str] = Field(description='Названия понятий, без которых это понятие не понять')
    misconceptions: list[str] = Field(description='1-2 типичных ошибочных представления новичков')


class SectionConcepts(BaseModel):
    concepts: list[ExtractedConcept] = Field(description='1-2 ключевых понятия раздела')


class Prerequisites(BaseModel):
    reason: str = Field(description='Коротко: на что опирается понятие')
    requires: list[str] = Field(description='1-2 названия из списка понятий, объяснённых раньше')


class GeneratedQuestion(BaseModel):
    kind: Literal['recall', 'apply', 'analyze']
    text: str = Field(description='Открытый вопрос без вариантов ответа')
    reference_answer: str = Field(description='Эталонный ответ, 2-4 предложения')
    key_points: list[str] = Field(description='2-4 коротких пункта, без которых ответ неполный')


class QuestionSet(BaseModel):
    questions: list[GeneratedQuestion]


EXTRACT_PROMPT = ChatPromptTemplate.from_messages([
    ('system',
     'Ты методист. Из раздела лекции выдели главное понятие - большую идею, которую студент должен понять, '
     'а не просто запомнить. Второе понятие добавляй, только если раздел явно объясняет две разные идеи. '
     'Пиши по-русски.\n'
     'Правила:\n'
     '- name: короткое название с заглавной буквы, как в тексте лекции;\n'
     '- свойство или деталь понятия - часть этого понятия, а не отдельное понятие '
     '(LIFO - свойство стека, декоратор - инструмент мемоизации);\n'
     '- не выделяй синтаксис, названия функций и общие слова вроде "ясность кода";\n'
     '- requires: только понятия этого курса - из уже найденных ({known}) или из этого же раздела;\n'
     '- misconceptions: реальные ошибки новичков именно по этой теме, одной фразой;\n'
     '- не добавляй фактов, которых нет в тексте.'),
    ('human', 'Курс: {course}\nРаздел: {section}\n\n{text}'),
])

LINK_PROMPT = ChatPromptTemplate.from_messages([
    ('system',
     'Ты методист курса «{course}». Студент собирается разобраться в понятии - не просто запомнить определение, '
     'а объяснить, как и почему оно работает. Какие 1-2 понятия из уже объяснённых ему нужно понимать заранее? '
     'Выбери самые прямые: те, на которые объяснение этого понятия опирается напрямую. '
     'Пустой список - только если понятие совсем не опирается на остальные. '
     'Не выбирай понятие только потому, что оно стоит последним в списке. Названия пиши слово в слово из списка. '
     'reason - одна короткая фраза по-русски.'),
    ('human', 'Понятие: {name} - {summary}\n\nОбъяснены раньше:\n{earlier}'),
])

QUESTION_PROMPT = ChatPromptTemplate.from_messages([
    ('system',
     'Ты составляешь проверочные вопросы для курса «{course}». По понятию напиши ровно 3 открытых вопроса:\n'
     '- recall: объяснить суть своими словами;\n'
     '- apply: маленькая задача или пример кода, где понятие надо применить;\n'
     '- analyze: почему так, сравни, найди ошибку - лучше всего через типичное заблуждение.\n'
     'Каждый вопрос проверяет именно это понятие, а не соседние темы из фрагментов. '
     'Вопросы и ответы опирай только на фрагменты лекции. Пиши по-русски, обращайся на "ты".\n\n'
     'Фрагменты лекции:\n{context}'),
    ('human', 'Понятие: {name}\nСуть: {summary}\nТипичные заблуждения: {misconceptions}'),
])


def normalize(name: str) -> str:
    return re.sub(r'[^\w ]', '', name.lower().replace('ё', 'е')).strip()


def find_same(name: str, names: list[str]) -> str | None:
    match = difflib.get_close_matches(normalize(name), [normalize(n) for n in names], n=1, cutoff=0.82)
    return next(n for n in names if normalize(n) == match[0]) if match else None


def merge(found: list[ExtractedConcept], new: list[ExtractedConcept]) -> list[ExtractedConcept]:
    """Одно понятие часто всплывает в нескольких разделах - склеиваем по похожему названию."""
    for concept in new:
        same = find_same(concept.name, [c.name for c in found])
        if not same:
            found.append(concept)
            continue
        target = next(c for c in found if c.name == same)
        target.requires = list(dict.fromkeys(target.requires + concept.requires))
        target.misconceptions = list(dict.fromkeys(target.misconceptions + concept.misconceptions))[:3]
    return found


def link_chain():
    return (LINK_PROMPT | llm.get_chat_model().with_structured_output(Prerequisites)).with_retry(stop_after_attempt=2)


async def link_concepts(course: Course, found: list[ExtractedConcept]) -> dict[str, list[str]]:
    """Связи - отдельным проходом, по маленькому вопросу на понятие: «что из уже объяснённого нужно, чтобы это понять».
    Одним запросом по всей карте 7B-модель то вытягивает её в цепочку по порядку лекции, то делает звезду.
    Предпосылки берутся только из того, что в лекции раньше, поэтому циклов не бывает. Если вопрос упал -
    остаются связи из первого прохода."""
    inputs = [{'course': course.title, 'name': c.name, 'summary': c.summary,
               'earlier': '\n'.join(f'- {e.name} - {e.summary}' for e in found[:i])}
              for i, c in enumerate(found) if i]
    results = await link_chain().abatch(inputs, config={**llm.run_config('forge.link', course_id=course.id),
                                                        'max_concurrency': 2}, return_exceptions=True)
    links = {found[0].name: []} if found else {}
    for i, (concept, result) in enumerate(zip(found[1:], results), start=1):
        earlier = [e.name for e in found[:i]]
        picked = concept.requires if isinstance(result, Exception) else result.requires
        links[concept.name] = [same for r in picked if (same := find_same(r, earlier))][:3]
    return links


def title_case(name: str) -> str:
    name = name.strip()
    return name[:1].upper() + name[1:] if re.match(r'[а-яё]', name) else name


def extract_chain():
    return (EXTRACT_PROMPT | llm.get_chat_model().with_structured_output(SectionConcepts)).with_retry(
        stop_after_attempt=2)


def question_chain(course_id: int):
    retriever = CourseRetriever(course_id=course_id, k=3, min_score=0.0)
    return (RunnableParallel(
        context=RunnableLambda(lambda x: f'{x["name"]}. {x["summary"]}') | retriever | format_docs,
        course=itemgetter('course'),
        name=itemgetter('name'),
        summary=itemgetter('summary'),
        misconceptions=lambda x: '; '.join(x['misconceptions']) or 'нет',
    ) | QUESTION_PROMPT | llm.get_chat_model(0.4).with_structured_output(QuestionSet)).with_retry(
        stop_after_attempt=2)


def set_progress(db, course: Course, text: str):
    course.progress = text
    db.commit()


async def build_course(course_id: int):
    db = SessionLocal()
    course = db.query(Course).filter(Course.id == course_id).first()
    try:
        set_progress(db, course, 'Читаю материалы')
        sections = split_sections(MaterialLoader(course.course_materials).load())
        if not sections:
            raise ValueError('В материалах нет текста')
        chunks = split_chunks(sections)
        set_progress(db, course, f'Индексирую {len(chunks)} фрагментов')
        await build_store(course.id, chunks)

        found: list[ExtractedConcept] = []
        sections_of: dict[str, str] = {}
        skipped = 0
        extract = extract_chain()
        for number, section in enumerate(sections, start=1):
            set_progress(db, course, f'Ищу понятия: раздел {number} из {len(sections)}')
            try:
                result = await extract.ainvoke(
                    {'course': course.title, 'section': section.metadata['section'], 'text': section.page_content,
                     'known': ', '.join(c.name for c in found) or 'пока нет'},
                    config=llm.run_config('forge.extract', course_id=course.id))
            except Exception:
                skipped += 1  # один неудачный раздел не должен валить весь курс
                continue
            for concept in result.concepts:
                concept.name = title_case(concept.name)
            found = merge(found, result.concepts[:2])
            for concept in result.concepts[:2]:
                canonical = find_same(concept.name, [c.name for c in found]) or concept.name
                sections_of.setdefault(canonical, section.metadata['section'])

        if not found:
            raise ValueError('Модель не нашла в материалах ни одного понятия')
        set_progress(db, course, f'Строю карту из {len(found)} понятий')
        links = await link_concepts(course, found)
        for old in list(course.course_concepts):
            db.delete(old)
        db.flush()
        rows = {}
        for concept in found:
            rows[concept.name] = Concept(course_id=course.id, name=concept.name, summary=concept.summary,
                                         section=sections_of.get(concept.name, ''),
                                         misconceptions=concept.misconceptions[:3])
            db.add(rows[concept.name])
        db.flush()
        names = list(rows)
        requires = {rows[c.name].id: {rows[same].id for r in links[c.name]
                                      if (same := find_same(r, names)) and same != c.name} for c in found}
        requires = graph.reduce_transitive(graph.drop_cycles(list(requires), requires))
        concept_depths = graph.depths(list(requires), requires)
        for concept in rows.values():
            concept.depth = concept_depths[concept.id]
            for parent in requires[concept.id]:
                db.add(ConceptLink(concept_id=concept.id, requires_id=parent))
        db.commit()

        set_progress(db, course, f'Пишу вопросы для {len(rows)} понятий')
        concepts = list(rows.values())
        sets = await question_chain(course.id).abatch(
            [{'course': course.title, 'name': c.name, 'summary': c.summary, 'misconceptions': c.misconceptions}
             for c in concepts],
            config={**llm.run_config('forge.questions', course_id=course.id), 'max_concurrency': 2},
            return_exceptions=True)
        total = 0
        for concept, question_set in zip(concepts, sets):
            if isinstance(question_set, Exception):
                continue
            for question in question_set.questions[:4]:
                db.add(Question(concept_id=concept.id, kind=QuestionKind(question.kind), text=question.text,
                                reference_answer=question.reference_answer, key_points=question.key_points[:4]))
                total += 1
        if not total:
            first = next((s for s in sets if isinstance(s, Exception)), None)
            raise ValueError(f'Модель не смогла составить ни одного вопроса: {first!r}')

        course.status = CourseStatus.review
        course.error = None
        empty = sum(1 for s in sets if isinstance(s, Exception))
        course.progress = (f'Готово: {len(rows)} {plural(len(rows), "понятие", "понятия", "понятий")}, '
                           f'{total} {plural(total, "вопрос", "вопроса", "вопросов")}') + (
            f', без вопросов {empty} понятий - добавь вручную' if empty else '') + (
            f', пропущено разделов: {skipped}' if skipped else '')
        db.commit()
    except Exception as e:
        db.rollback()
        course.status = CourseStatus.failed
        course.error = f'{type(e).__name__}: {e}'
        db.commit()
    finally:
        db.close()
