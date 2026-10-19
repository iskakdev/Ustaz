from collections import Counter, defaultdict
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import case, func
from sqlalchemy.orm import Session
from ai.insight import build_brief
from database.db import get_db
from database.models import AIUsage, Attempt, Concept, Mastery, Question, UserProfile, Verdict
from api.auth import get_teacher
from api.course import own_course
from mastery import bkt
from mastery.state import course_graph

insight_router = APIRouter(prefix='/insight', tags=['Insight'])


def concept_stats(course_id: int, db: Session) -> list[dict]:
    concepts, requires = course_graph(db, course_id)
    attempts = db.query(Attempt).join(Question).filter(Question.concept_id.in_(concepts)).all()
    by_concept = defaultdict(list)
    for attempt in attempts:
        by_concept[attempt.attempt_question.concept_id].append(attempt)
    masteries = defaultdict(list)
    for m in db.query(Mastery).filter(Mastery.concept_id.in_(concepts)).all():
        masteries[m.concept_id].append(m.p_known)

    stats = []
    for cid, concept in sorted(concepts.items(), key=lambda item: (item[1].depth, item[0])):
        rows = by_concept[cid]
        stats.append({
            'id': cid, 'name': concept.name, 'depth': concept.depth,
            'requires': [concepts[r].name for r in sorted(requires[cid])],
            'students': len(masteries[cid]),
            'mastered': sum(1 for p in masteries[cid] if p >= bkt.MASTERED),
            'avg_mastery': round(sum(masteries[cid]) / len(masteries[cid]), 2) if masteries[cid] else None,
            'attempts': len(rows),
            'avg_score': round(sum(a.score for a in rows) / len(rows), 2) if rows else None,
            'top_misconceptions': Counter(a.misconception for a in rows if a.misconception).most_common(3),
            'wrong': [a for a in rows if a.verdict != Verdict.correct],
            'summary': concept.summary,
        })
    return stats


@insight_router.get('/{course_id}/concepts/', response_model=list)
async def concepts(course_id: int, db: Session = Depends(get_db), teacher: UserProfile = Depends(get_teacher)):
    """Где группа спотыкается: средний балл, доля освоивших и частые заблуждения по каждому понятию."""
    own_course(course_id, teacher, db)
    return [{k: v for k, v in s.items() if k not in ('wrong', 'summary')} for s in concept_stats(course_id, db)]


@insight_router.post('/{course_id}/brief/', response_model=dict)
async def brief(course_id: int, db: Session = Depends(get_db), teacher: UserProfile = Depends(get_teacher)):
    """План следующего занятия по реальным ошибкам студентов."""
    course_db = own_course(course_id, teacher, db)
    stats = concept_stats(course_id, db)
    struggles = sorted((s for s in stats if len(s['wrong']) >= 1), key=lambda s: s['avg_score'])[:4]
    if not struggles:
        raise HTTPException(detail='Пока нет неверных ответов - разбирать нечего', status_code=400)

    result = await build_brief(course_db.title, [{
        'concept': s['name'], 'summary': s['summary'], 'avg_score': s['avg_score'],
        'misconceptions': '; '.join(f'{m} ({n})' for m, n in s['top_misconceptions']) or 'не выявлены',
        'answers': '\n'.join(f'- ({a.score:.0%}) {a.answer[:300]}' for a in s['wrong'][-8:]),
    } for s in struggles], '\n'.join(f'{s["name"]} <- {", ".join(s["requires"]) or "-"}' for s in stats))
    return result.model_dump()


@insight_router.get('/usage/', response_model=list)
async def usage(db: Session = Depends(get_db), teacher: UserProfile = Depends(get_teacher)):
    """Расход модели по фичам из колбэка UsageCallback: вызовы, кэш, ошибки, токены, задержка."""
    rows = db.query(AIUsage.feature, func.count(AIUsage.id), func.sum(case((AIUsage.cached.is_(True), 1), else_=0)),
                    func.sum(case((AIUsage.ok.is_(False), 1), else_=0)), func.sum(AIUsage.input_tokens),
                    func.sum(AIUsage.output_tokens), func.avg(AIUsage.latency_ms)).group_by(AIUsage.feature).all()
    return [{'feature': f, 'calls': calls, 'cached': cached or 0, 'errors': errors or 0, 'input_tokens': inp or 0,
             'output_tokens': out or 0, 'avg_latency_ms': round(lat or 0)} for f, calls, cached, errors, inp, out, lat
            in rows]
