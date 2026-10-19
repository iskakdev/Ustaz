"""Что спросить дальше. Порядок: диагностика -> повторение того, что выцветает -> новое понятие."""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from mastery import bkt, graph


@dataclass
class ConceptState:
    id: int
    name: str
    depth: int
    p: float
    stability: float
    attempts: int
    last_seen: Optional[datetime]


@dataclass
class Step:
    mode: str  # diagnostic | review | learn | done
    concept_id: Optional[int]
    kind: Optional[str]
    reason: str


def plural(n: int, one: str, few: str, many: str) -> str:
    """1 понятие, 2 понятия, 5 понятий."""
    if n % 10 == 1 and n % 100 != 11:
        return one
    return few if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14 else many


def kind_for(p: float) -> str:
    if p < 0.45:
        return 'recall'
    if p < 0.75:
        return 'apply'
    return 'analyze'


def pick_next(states: dict[int, ConceptState], requires: graph.Requires, now: datetime,
              diagnostic_left: int) -> Step:
    step = (diagnostic(states, requires, diagnostic_left)
            or review(states, now)
            or learn(states, requires))
    if step:
        return step
    upcoming = [bkt.next_review(s.last_seen, s.stability) for s in states.values() if s.last_seen]
    when = f' Следующее повторение после {min(upcoming):%d.%m %H:%M} UTC.' if upcoming else ''
    return Step('done', None, None, 'Курс освоен, сейчас повторять нечего.' + when)


def diagnostic(states: dict[int, ConceptState], requires: graph.Requires, left: int) -> Optional[Step]:
    """Бинарный поиск по карте: спрашиваем самое глубокое непроверенное понятие.
    Ответил - зачитываем всю цепочку предпосылок, не ответил - спускаемся ниже."""
    if left <= 0:
        return None
    failed = [s.id for s in states.values() if s.attempts and s.p < bkt.UNLOCK]
    blocked = set().union(*(graph.descendants(f, requires) for f in failed)) if failed else set()
    candidates = [s for s in states.values() if s.last_seen is None and s.id not in blocked]
    if not candidates:
        return None
    target = max(candidates, key=lambda s: (s.depth, len(graph.ancestors(s.id, requires)), -s.id))
    base = sum(1 for a in graph.ancestors(target.id, requires) if states[a].p < bkt.MASTERED)  # уже зачтённые не считаем
    noun = plural(base, 'базовое понятие', 'базовых понятия', 'базовых понятий')
    reason = (f'Диагностика: сразу проверяю «{target.name}». Справишься - зачту ещё {base} {noun} без вопросов.'
              if base else f'Диагностика: проверяю, знаком ли ты с «{target.name}».')
    return Step('diagnostic', target.id, 'analyze', reason)


def review(states: dict[int, ConceptState], now: datetime) -> Optional[Step]:
    due = [s for s in states.values()
           if s.last_seen and s.p >= bkt.UNLOCK and now >= bkt.next_review(s.last_seen, s.stability)]
    if not due:
        return None
    target = min(due, key=lambda s: bkt.next_review(s.last_seen, s.stability))
    left = bkt.retention(target.stability, target.last_seen, now)
    reason = f'Повторение: по кривой забывания ты сейчас помнишь «{target.name}» примерно на {left:.0%}.'
    return Step('review', target.id, kind_for(target.p), reason)


def learn(states: dict[int, ConceptState], requires: graph.Requires) -> Optional[Step]:
    frontier = [s for s in states.values()
                if s.p < bkt.MASTERED and all(states[r].p >= bkt.UNLOCK for r in requires.get(s.id, ()) if r in states)]
    if not frontier:
        return None

    def priority(s: ConceptState):
        stuck = s.attempts >= 4 and s.p < bkt.UNLOCK  # застрял - даём передышку и чередуем с другим
        return stuck, -len(graph.descendants(s.id, requires)), s.depth, s.id

    target = min(frontier, key=priority)
    unlocks = [states[d].name for d in graph.descendants(target.id, requires) if d in states]
    if target.attempts:
        reason = f'Продолжаем «{target.name}»: сейчас {target.p:.0%}, освоенным считаю от {bkt.MASTERED:.0%}.'
    elif unlocks:
        reason = f'Новое понятие «{target.name}»: откроет ещё {len(unlocks)} - {", ".join(sorted(unlocks)[:3])}.'
    else:
        reason = f'Новое понятие «{target.name}»: все предпосылки у тебя уже есть.'
    return Step('learn', target.id, kind_for(target.p), reason)
