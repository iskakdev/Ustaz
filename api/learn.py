import difflib
from collections import Counter
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ai import grader
from ai.forge import normalize
from database.db import get_db, utcnow
from database.models import (Attempt, Course, CourseStatus, Enrollment, Question, QuestionKind, UserProfile,
                             UserRole, Verdict)
from database.schema import AnswerInputSchema
from api.auth import get_current_user
from mastery import bkt, graph
from mastery.planner import pick_next
from mastery.state import get_mastery, mermaid, status, student_states

learn_router = APIRouter(prefix='/learn', tags=['Learn'])

KIND_ORDER = ['recall', 'apply', 'analyze']


def published_course(course_id: int, db: Session) -> Course:
    course_db = db.query(Course).filter(Course.id == course_id, Course.status == CourseStatus.published).first()
    if not course_db:
        raise HTTPException(detail='Курс не найден или ещё не опубликован', status_code=404)
    return course_db


def get_enrollment(course_id: int, user: UserProfile, db: Session) -> Enrollment:
    published_course(course_id, db)
    enrollment = db.query(Enrollment).filter(Enrollment.course_id == course_id,
                                             Enrollment.student_id == user.id).first()
    if not enrollment:
        raise HTTPException(detail='Сначала запишись на курс: POST /learn/{course_id}/enroll/', status_code=400)
    return enrollment


def pick_question(db: Session, student_id: int, concept_id: int, kind: str) -> Question:
    """Ближайший по уровню вопрос, но не тот, что был в прошлый раз, и пореже отвеченный."""
    questions = db.query(Question).filter(Question.concept_id == concept_id, Question.is_active.is_(True)).all()
    answered = Counter(qid for (qid,) in db.query(Attempt.question_id).filter(
        Attempt.student_id == student_id, Attempt.question_id.in_([q.id for q in questions])))
    last = get_mastery(db, student_id, concept_id).last_question_id
    return min(questions, key=lambda q: (q.id == last and len(questions) > 1,
                                         abs(KIND_ORDER.index(q.kind.value) - KIND_ORDER.index(kind)),
                                         answered[q.id], q.id))


def open_concepts(states, requires) -> set[int]:
    return {cid for cid, s in states.items() if s.p < bkt.MASTERED
            and all(states[r].p >= bkt.UNLOCK for r in requires.get(cid, ()))}


def canonical(misconception: Optional[str], known: list[str]) -> Optional[str]:
    """Сводим формулировку модели к известному заблуждению, чтобы аналитика считала их вместе."""
    if not misconception or normalize(misconception) in ('', 'null', 'none', 'нет'):
        return None
    best = max(known, key=lambda k: difflib.SequenceMatcher(None, normalize(k), normalize(misconception)).ratio(),
               default=None)
    if best and difflib.SequenceMatcher(None, normalize(best), normalize(misconception)).ratio() >= 0.6:
        return best
    return misconception[:200]


def question_out(question: Question, mode: str, reason: str, p: float) -> dict:
    return {'mode': mode, 'reason': reason,
            'concept': {'id': question.concept_id, 'name': question.question_concept.name, 'mastery': round(p, 2)},
            'question': {'id': question.id, 'kind': question.kind.value, 'text': question.text}}


@learn_router.post('/{course_id}/enroll/', response_model=dict)
async def enroll(course_id: int, db: Session = Depends(get_db),
                 current_user: UserProfile = Depends(get_current_user)):
    if current_user.role != UserRole.student:
        raise HTTPException(detail='Записаться может только студент', status_code=403)
    course_db = published_course(course_id, db)
    if db.query(Enrollment).filter(Enrollment.course_id == course_id, Enrollment.student_id == current_user.id).first():
        raise HTTPException(detail='Ты уже записан на этот курс', status_code=400)

    db.add(Enrollment(course_id=course_id, student_id=current_user.id))
    db.commit()
    return {'message': f'Ты записан на «{course_db.title}». Начни с GET /learn/{course_id}/next/ - '
                       f'первые вопросы покажут, что ты уже знаешь.'}


@learn_router.get('/{course_id}/next/', response_model=dict)
async def next_step(course_id: int, db: Session = Depends(get_db),
                    current_user: UserProfile = Depends(get_current_user)):
    enrollment = get_enrollment(course_id, current_user, db)
    states, requires, _ = student_states(db, current_user.id, course_id)
    if enrollment.open_question:
        question = enrollment.open_question
        return question_out(question, enrollment.open_mode, 'Сначала ответь на открытый вопрос.',
                            states[question.concept_id].p if question.concept_id in states else bkt.P_INIT)

    step = pick_next(states, requires, utcnow(), enrollment.diagnostic_left)
    if step.mode == 'done':
        return {'mode': 'done', 'reason': step.reason}

    question = pick_question(db, current_user.id, step.concept_id, step.kind)
    enrollment.open_question_id = question.id
    enrollment.open_mode = step.mode
    db.commit()
    return question_out(question, step.mode, step.reason, states[step.concept_id].p)


@learn_router.post('/{course_id}/answer/', response_model=dict)
async def answer(course_id: int, data: AnswerInputSchema, db: Session = Depends(get_db),
                 current_user: UserProfile = Depends(get_current_user)):
    enrollment = get_enrollment(course_id, current_user, db)
    if enrollment.open_question_id != data.question_id:
        raise HTTPException(detail=f'Этот вопрос сейчас не открыт. Возьми вопрос: GET /learn/{course_id}/next/',
                            status_code=400)

    question = enrollment.open_question
    concept = question.question_concept
    mode = enrollment.open_mode
    grade, graded_by = await grader.grade(enrollment.enrollment_course.title, concept.name, question.text,
                                          question.reference_answer, question.key_points, concept.misconceptions,
                                          data.answer)
    score = round(grade.score, 2)
    verdict = Verdict.correct if score >= 0.75 else Verdict.partial if score >= 0.4 else Verdict.wrong
    misconception = canonical(grade.misconception, concept.misconceptions)

    now = utcnow()
    states, requires, concepts = student_states(db, current_user.id, course_id)
    was_open = open_concepts(states, requires)

    mastery = get_mastery(db, current_user.id, concept.id)
    before = mastery.p_known
    mastery.p_known = bkt.update(before, score, guess=bkt.GUESS_BY_KIND[question.kind.value])
    mastery.stability = bkt.next_stability(mastery.stability, score)
    mastery.attempts += 1
    mastery.last_seen_at = now
    mastery.last_question_id = question.id

    credited = []
    placed = mode == 'diagnostic' and score >= 0.75
    if placed or (score >= 0.7 and question.kind != QuestionKind.recall):
        for cid in graph.ancestors(concept.id, requires) | ({concept.id} if placed else set()):
            ancestor = get_mastery(db, current_user.id, cid)
            new_p = max(ancestor.p_known, bkt.PLACED) if placed else bkt.credit(ancestor.p_known, score)
            if new_p - ancestor.p_known >= 0.01 and cid != concept.id:
                credited.append(concepts[cid].name)
            ancestor.p_known = new_p
            if ancestor.last_seen_at is None or placed:
                ancestor.last_seen_at = now
                ancestor.stability = max(ancestor.stability, bkt.PLACED_STABILITY)
    if mode == 'diagnostic':
        enrollment.diagnostic_left -= 1

    db.add(Attempt(student_id=current_user.id, question_id=question.id, mode=mode, answer=data.answer, score=score,
                   verdict=verdict, feedback=grade.feedback, misconception=misconception, graded_by=graded_by))
    enrollment.open_question_id = None
    db.commit()

    states, requires, _ = student_states(db, current_user.id, course_id)
    unlocked = sorted(states[cid].name for cid in open_concepts(states, requires) - was_open)
    return {
        'verdict': verdict.value,
        'score': score,
        'graded_by': graded_by,
        'feedback': grade.feedback,
        'covered': grade.covered,
        'missing': grade.missing,
        'misconception': misconception,
        'reference_answer': question.reference_answer,
        'mastery': {'concept': concept.name, 'before': round(before, 2), 'after': round(mastery.p_known, 2),
                    'status': status(states[concept.id], requires, states, now)},
        'credited': sorted(credited),
        'unlocked': unlocked,
        'next_review_at': bkt.next_review(now, mastery.stability).isoformat(timespec='minutes'),
    }


@learn_router.get('/{course_id}/progress/', response_model=dict)
async def progress(course_id: int, db: Session = Depends(get_db),
                   current_user: UserProfile = Depends(get_current_user)):
    enrollment = get_enrollment(course_id, current_user, db)
    states, requires, _ = student_states(db, current_user.id, course_id)
    now = utcnow()
    items = []
    for s in sorted(states.values(), key=lambda s: (s.depth, s.id)):
        items.append({'id': s.id, 'name': s.name, 'depth': s.depth, 'p_known': round(s.p, 2),
                      'remembered_now': round(bkt.effective(s.p, s.stability, s.last_seen, now), 2),
                      'status': status(s, requires, states, now), 'attempts': s.attempts,
                      'next_review_at': bkt.next_review(s.last_seen, s.stability).isoformat(timespec='minutes')
                      if s.last_seen else None})
    mastered = sum(1 for s in states.values() if s.p >= bkt.MASTERED)
    return {'course': enrollment.enrollment_course.title, 'mastered': mastered, 'total': len(states),
            'percent': round(100 * mastered / len(states)) if states else 0,
            'diagnostic_left': enrollment.diagnostic_left,
            'due_now': [i['name'] for i in items if i['status'] == 'fading'], 'concepts': items}


@learn_router.get('/{course_id}/map/', response_model=dict)
async def my_map(course_id: int, db: Session = Depends(get_db),
                 current_user: UserProfile = Depends(get_current_user)):
    get_enrollment(course_id, current_user, db)
    states, requires, _ = student_states(db, current_user.id, course_id)
    now = utcnow()
    return {'mermaid': mermaid({cid: s.name for cid, s in states.items()}, requires,
                               labels={cid: f'{s.p:.0%}' for cid, s in states.items()},
                               statuses={cid: status(s, requires, states, now) for cid, s in states.items()}),
            'legend': {'mastered': 'освоено', 'learning': 'в процессе', 'fading': 'пора повторить',
                       'open': 'можно начинать', 'locked': 'сначала освой предпосылки'}}
