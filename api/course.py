from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile
from sqlalchemy import or_
from sqlalchemy.orm import Session
from typing import List
from ai.forge import build_course
from database.db import get_db
from database.models import Course, CourseStatus, Material, Question, Concept, ConceptLink, UserProfile
from database.schema import (CourseInputSchema, CourseOutSchema, MaterialInputSchema, MaterialOutSchema,
                             QuestionOutSchema, QuestionInputSchema, QuestionUpdateSchema, ConceptUpdateSchema)
from api.auth import get_current_user, get_teacher
from mastery import graph
from mastery.state import course_graph, mermaid

course_router = APIRouter(prefix='/courses', tags=['Courses'])

EDITABLE = (CourseStatus.draft, CourseStatus.review, CourseStatus.failed)


def own_course(course_id: int, teacher: UserProfile, db: Session) -> Course:
    course_db = db.query(Course).filter(Course.id == course_id).first()
    if not course_db:
        raise HTTPException(detail='Курс не найден', status_code=404)
    if course_db.teacher_id != teacher.id:
        raise HTTPException(detail='Это не ваш курс', status_code=403)
    return course_db


def add_material(course_db: Course, title: str, text: str, db: Session) -> Material:
    if course_db.status not in EDITABLE:
        raise HTTPException(detail='Курс уже собирается или опубликован', status_code=400)
    material_db = Material(course_id=course_db.id, title=title, text=text)
    db.add(material_db)
    course_db.status = CourseStatus.draft
    db.commit()
    db.refresh(material_db)
    return material_db


@course_router.post('/', response_model=CourseOutSchema)
async def create_course(course: CourseInputSchema, db: Session = Depends(get_db),
                        teacher: UserProfile = Depends(get_teacher)):
    course_db = Course(**course.model_dump(), teacher_id=teacher.id)
    db.add(course_db)
    db.commit()
    db.refresh(course_db)
    return course_db


@course_router.get('/', response_model=List[CourseOutSchema])
async def list_course(db: Session = Depends(get_db), current_user: UserProfile = Depends(get_current_user)):
    """Студент видит опубликованные курсы, преподаватель - ещё и свои черновики."""
    return db.query(Course).filter(or_(Course.status == CourseStatus.published,
                                       Course.teacher_id == current_user.id)).all()


@course_router.get('/{course_id}/', response_model=CourseOutSchema)
async def detail_course(course_id: int, db: Session = Depends(get_db),
                        current_user: UserProfile = Depends(get_current_user)):
    course_db = db.query(Course).filter(Course.id == course_id).first()
    if not course_db or (course_db.status != CourseStatus.published and course_db.teacher_id != current_user.id):
        raise HTTPException(detail='Курс не найден', status_code=404)
    return course_db


@course_router.post('/{course_id}/materials/', response_model=MaterialOutSchema)
async def create_material(course_id: int, material: MaterialInputSchema, db: Session = Depends(get_db),
                          teacher: UserProfile = Depends(get_teacher)):
    return add_material(own_course(course_id, teacher, db), material.title, material.text, db)


@course_router.post('/{course_id}/materials/upload/', response_model=MaterialOutSchema)
async def upload_material(course_id: int, file: UploadFile, db: Session = Depends(get_db),
                          teacher: UserProfile = Depends(get_teacher)):
    if not (file.filename or '').lower().endswith(('.md', '.txt')):
        raise HTTPException(detail='Нужен .md или .txt', status_code=400)
    try:
        text = (await file.read()).decode('utf-8')
    except UnicodeDecodeError:
        raise HTTPException(detail='Файл должен быть в UTF-8', status_code=400)
    return add_material(own_course(course_id, teacher, db), file.filename, text, db)


@course_router.get('/{course_id}/materials/', response_model=List[MaterialOutSchema])
async def list_material(course_id: int, db: Session = Depends(get_db), teacher: UserProfile = Depends(get_teacher)):
    return own_course(course_id, teacher, db).course_materials


@course_router.post('/{course_id}/build/', response_model=dict, status_code=202)
async def build(course_id: int, background: BackgroundTasks, db: Session = Depends(get_db),
                teacher: UserProfile = Depends(get_teacher)):
    course_db = own_course(course_id, teacher, db)
    if course_db.status not in EDITABLE:
        raise HTTPException(detail='Курс уже собирается или опубликован', status_code=400)
    if not course_db.course_materials:
        raise HTTPException(detail='Сначала добавьте материалы', status_code=400)

    course_db.status = CourseStatus.building
    course_db.progress = 'В очереди'
    db.commit()
    background.add_task(build_course, course_db.id)
    return {'message': 'Собираю курс. Следите за status и progress в GET /courses/{id}/', 'status': 'building'}


@course_router.get('/{course_id}/map/', response_model=dict)
async def course_map(course_id: int, db: Session = Depends(get_db), teacher: UserProfile = Depends(get_teacher)):
    own_course(course_id, teacher, db)
    concepts, requires = course_graph(db, course_id)
    counts = {c.id: sum(1 for q in c.concept_questions if q.is_active) for c in concepts.values()}
    return {
        'concepts': [{'id': c.id, 'name': c.name, 'summary': c.summary, 'section': c.section, 'depth': c.depth,
                      'requires': sorted(requires[c.id]), 'misconceptions': c.misconceptions,
                      'questions': counts[c.id]} for c in sorted(concepts.values(), key=lambda c: (c.depth, c.id))],
        'mermaid': mermaid({cid: c.name for cid, c in concepts.items()}, requires,
                           labels={cid: f'{n} вопр.' for cid, n in counts.items()}),
    }


@course_router.put('/concepts/{concept_id}/', response_model=dict)
async def update_concept(concept_id: int, concept: ConceptUpdateSchema, db: Session = Depends(get_db),
                         teacher: UserProfile = Depends(get_teacher)):
    """Преподаватель правит карту за моделью: название, суть, заблуждения и предпосылки (id понятий)."""
    concept_db = db.query(Concept).filter(Concept.id == concept_id).first()
    if not concept_db:
        raise HTTPException(detail='Понятие не найдено', status_code=404)
    course_db = own_course(concept_db.course_id, teacher, db)
    if course_db.status not in EDITABLE:
        raise HTTPException(detail='Карту опубликованного курса менять нельзя: собьётся прогресс студентов',
                            status_code=400)

    data = concept.model_dump(exclude_unset=True)
    requires = data.pop('requires', None)
    for key, value in data.items():
        setattr(concept_db, key, value)
    if requires is not None:
        concepts, links = course_graph(db, course_db.id)
        if concept_id in requires or set(requires) - set(concepts):
            raise HTTPException(detail='Предпосылки - это id других понятий этого курса', status_code=400)
        links[concept_id] = set(requires)
        if graph.drop_cycles(list(links), links) != links:
            raise HTTPException(detail='Получится цикл: понятие стало бы предпосылкой самого себя', status_code=400)
        db.query(ConceptLink).filter(ConceptLink.concept_id == concept_id).delete()
        db.add_all([ConceptLink(concept_id=concept_id, requires_id=r) for r in requires])
        for cid, depth in graph.depths(list(links), links).items():
            concepts[cid].depth = depth
    db.commit()
    return {'message': 'Понятие изменено', 'depth': concept_db.depth}


@course_router.post('/concepts/{concept_id}/questions/', response_model=QuestionOutSchema)
async def create_question(concept_id: int, question: QuestionInputSchema, db: Session = Depends(get_db),
                          teacher: UserProfile = Depends(get_teacher)):
    concept_db = db.query(Concept).filter(Concept.id == concept_id).first()
    if not concept_db:
        raise HTTPException(detail='Понятие не найдено', status_code=404)
    own_course(concept_db.course_id, teacher, db)

    question_db = Question(concept_id=concept_id, **question.model_dump())
    db.add(question_db)
    db.commit()
    db.refresh(question_db)
    return question_db


@course_router.get('/{course_id}/questions/', response_model=List[QuestionOutSchema])
async def list_question(course_id: int, db: Session = Depends(get_db), teacher: UserProfile = Depends(get_teacher)):
    own_course(course_id, teacher, db)
    return db.query(Question).join(Concept).filter(Concept.course_id == course_id).order_by(Question.id).all()


@course_router.put('/questions/{question_id}/', response_model=QuestionOutSchema)
async def update_question(question_id: int, question: QuestionUpdateSchema, db: Session = Depends(get_db),
                          teacher: UserProfile = Depends(get_teacher)):
    question_db = db.query(Question).filter(Question.id == question_id).first()
    if not question_db:
        raise HTTPException(detail='Вопрос не найден', status_code=404)
    own_course(question_db.question_concept.course_id, teacher, db)

    for key, value in question.model_dump(exclude_unset=True).items():
        setattr(question_db, key, value)
    db.commit()
    db.refresh(question_db)
    return question_db


@course_router.post('/{course_id}/publish/', response_model=dict)
async def publish(course_id: int, db: Session = Depends(get_db), teacher: UserProfile = Depends(get_teacher)):
    course_db = own_course(course_id, teacher, db)
    if course_db.status != CourseStatus.review:
        raise HTTPException(detail='Опубликовать можно только собранный курс', status_code=400)

    course_db.status = CourseStatus.published
    db.commit()
    return {'message': 'Курс опубликован'}
