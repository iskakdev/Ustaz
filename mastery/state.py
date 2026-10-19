"""Чтение карты и знаний студента из БД - общее для API, агента-коуча и аналитики."""
from datetime import datetime
from typing import Optional
from sqlalchemy.orm import Session
from database.models import Concept, ConceptLink, Mastery, Question
from mastery import bkt, graph
from mastery.planner import ConceptState


def course_graph(db: Session, course_id: int) -> tuple[dict[int, Concept], graph.Requires]:
    concepts = {c.id: c for c in db.query(Concept).filter(Concept.course_id == course_id).all()}
    requires: graph.Requires = {cid: set() for cid in concepts}
    links = db.query(ConceptLink).filter(ConceptLink.concept_id.in_(concepts)).all()
    for link in links:
        requires[link.concept_id].add(link.requires_id)
    return concepts, requires


def masteries(db: Session, student_id: int, concept_ids) -> dict[int, Mastery]:
    rows = db.query(Mastery).filter(Mastery.student_id == student_id, Mastery.concept_id.in_(concept_ids)).all()
    return {m.concept_id: m for m in rows}


def get_mastery(db: Session, student_id: int, concept_id: int) -> Mastery:
    mastery = db.query(Mastery).filter(Mastery.student_id == student_id, Mastery.concept_id == concept_id).first()
    if not mastery:
        mastery = Mastery(student_id=student_id, concept_id=concept_id, p_known=bkt.P_INIT, stability=1.0, attempts=0)
        db.add(mastery)
        db.flush()
    return mastery


def student_states(db: Session, student_id: int, course_id: int) -> tuple[dict[int, ConceptState], graph.Requires,
                                                                          dict[int, Concept]]:
    """Состояния только тех понятий, где есть активные вопросы: без вопроса понятие не проверить."""
    concepts, requires = course_graph(db, course_id)
    with_questions = {cid for (cid,) in db.query(Question.concept_id).filter(
        Question.concept_id.in_(concepts), Question.is_active.is_(True)).distinct()}
    rows = masteries(db, student_id, concepts)
    states = {}
    for cid, concept in concepts.items():
        if cid not in with_questions:
            continue
        m: Optional[Mastery] = rows.get(cid)
        states[cid] = ConceptState(id=cid, name=concept.name, depth=concept.depth,
                                   p=m.p_known if m else bkt.P_INIT, stability=m.stability if m else 1.0,
                                   attempts=m.attempts if m else 0, last_seen=m.last_seen_at if m else None)
    requires = {cid: {r for r in parents if r in states} for cid, parents in requires.items() if cid in states}
    return states, requires, concepts


def status(state: ConceptState, requires: graph.Requires, states: dict[int, ConceptState], now: datetime) -> str:
    if any(states[r].p < bkt.UNLOCK for r in requires.get(state.id, ())):
        return 'locked'
    if state.last_seen and state.p >= bkt.UNLOCK \
            and now >= bkt.next_review(state.last_seen, state.stability):
        return 'fading'
    if state.p >= bkt.MASTERED:
        return 'mastered'
    return 'learning' if state.attempts else 'open'


STYLE = {
    'mastered': 'fill:#2e7d32,color:#fff,stroke:#1b5e20',
    'learning': 'fill:#f9a825,color:#000,stroke:#f57f17',
    'fading': 'fill:#ef6c00,color:#fff,stroke:#e65100',
    'open': 'fill:#1565c0,color:#fff,stroke:#0d47a1',
    'locked': 'fill:#9e9e9e,color:#fff,stroke:#616161',
    'concept': 'fill:#eceff1,color:#000,stroke:#607d8b',
}


def mermaid(names: dict[int, str], requires: graph.Requires, labels: Optional[dict[int, str]] = None,
            statuses: Optional[dict[int, str]] = None) -> str:
    """Карту можно вставить в mermaid.live или в README на GitHub - она нарисуется."""
    lines = ['flowchart TD']
    used = set()
    for cid, name in names.items():
        label = name.replace('"', "'") + (f'<br/>{labels[cid]}' if labels and cid in labels else '')
        css = (statuses or {}).get(cid, 'concept')
        used.add(css)
        lines.append(f'  c{cid}["{label}"]:::{css}')
    for cid, parents in requires.items():
        for parent in sorted(parents):
            lines.append(f'  c{parent} --> c{cid}')
    lines += [f'  classDef {name} {style}' for name, style in STYLE.items() if name in used]
    return '\n'.join(lines)
