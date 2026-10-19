"""Bayesian Knowledge Tracing с мягкой оценкой и кривой забывания.

Обычный BKT получает ответ "верно/неверно". У нас ответ свободный, грейдер ставит score от 0 до 1,
поэтому апостериорная вероятность считается как смесь двух байесовских апдейтов с весом score.
Поверх этого живёт память: знание со временем выцветает по экспоненте, а каждое удачное
повторение делает её устойчивее (как в интервальных повторениях).
"""
import math
from datetime import datetime, timedelta
from typing import Optional

P_INIT = 0.2             # до первой попытки считаем, что понятие знакомо с вероятностью 20%
P_LEARN = 0.15           # шанс понять понятие за одну попытку с разбором ошибки
P_SLIP = 0.1             # знает, но ответил плохо
P_GUESS = 0.2            # не знает, но набрал нужных слов
GUESS_BY_KIND = {'recall': 0.25, 'apply': 0.15, 'analyze': 0.1}  # на сложном вопросе угадать труднее
MASTERED = 0.85          # понятие освоено
UNLOCK = 0.6             # можно переходить к понятиям, которые от него зависят
RETENTION_TARGET = 0.75  # повторяем, когда по кривой забывания помнишь 75%
CREDIT = 0.8             # во сколько раз слабее свидетельство для предпосылок
PLACED = 0.9             # диагностика пройдена: понятие и его база считаются освоенными
PLACED_STABILITY = 2.0   # но через ~14 часов проверим, не забылось ли


def posterior(p: float, score: float, guess: float = P_GUESS) -> float:
    p_correct = p * (1 - P_SLIP) + (1 - p) * guess
    known_if_correct = p * (1 - P_SLIP) / p_correct
    known_if_wrong = p * P_SLIP / (1 - p_correct)
    return score * known_if_correct + (1 - score) * known_if_wrong


def update(p: float, score: float, guess: float = P_GUESS) -> float:
    p = posterior(p, score, guess)
    return p + (1 - p) * P_LEARN


def credit(p: float, score: float) -> float:
    """Справился с задачей на зависимое понятие - значит, скорее всего знаешь и базу.
    Свидетельство слабее прямого (CREDIT), зато случайно применить незнакомую базу почти нельзя. Только повышает."""
    return max(p, update(p, score * CREDIT, guess=0.05))


def retention(stability: float, last_seen: Optional[datetime], now: datetime) -> float:
    if last_seen is None:
        return 1.0
    days = max(0.0, (now - last_seen).total_seconds() / 86400)
    return math.exp(-days / stability)


def effective(p: float, stability: float, last_seen: Optional[datetime], now: datetime) -> float:
    return p * retention(stability, last_seen, now)


def next_stability(stability: float, score: float) -> float:
    if score >= 0.7:
        return stability * (1.5 + score)
    if score < 0.4:
        return max(0.5, stability * 0.5)
    return stability


def next_review(last_seen: datetime, stability: float) -> datetime:
    return last_seen + timedelta(days=-stability * math.log(RETENTION_TARGET))
