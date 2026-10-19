"""Демо-данные: преподаватель, студент и курс «Рекурсия в Python», собранный ИИ из materials/recursion.md.

    .venv/bin/python seed_demo.py

Сборка идёт через локальную модель и на CPU занимает 10-20 минут. Повторный запуск пересобирает курс,
неизменённые промпты берутся из SQL-кэша и проходят мгновенно.
"""
import asyncio
import os
import time
from pathlib import Path
import main  # noqa: F401  создаёт таблицы и включает кэш
from ai.forge import build_course
from ai.knowledge import store_path
from api.auth import get_password_hash
from database.db import SessionLocal
from database.models import Course, CourseStatus, Material, UserProfile, UserRole
from mastery.state import course_graph, mermaid

USERS = [('Айгуль Преподаватель', 'teacher@ustaz.kg', 'teacher123', UserRole.teacher),
         ('Бакыт Студент', 'student@ustaz.kg', 'student123', UserRole.student)]
TITLE = 'Рекурсия в Python'


def get_user(db, full_name, email, password, role) -> UserProfile:
    user = db.query(UserProfile).filter(UserProfile.email == email).first()
    if not user:
        user = UserProfile(full_name=full_name, email=email, password=get_password_hash(password), role=role)
        db.add(user)
        db.commit()
    return user


async def seed():
    db = SessionLocal()
    teacher, _ = [get_user(db, *u) for u in USERS]
    for old in db.query(Course).filter(Course.title == TITLE, Course.teacher_id == teacher.id).all():
        if os.path.exists(store_path(old.id)):
            os.remove(store_path(old.id))
        db.delete(old)
    course = Course(title=TITLE, description='Стек вызовов, базовый случай, мемоизация и когда рекурсия не нужна',
                    teacher_id=teacher.id, status=CourseStatus.building)
    db.add(course)
    db.flush()
    db.add(Material(course_id=course.id, title='recursion.md',
                    text=(Path(__file__).parent / 'materials' / 'recursion.md').read_text(encoding='utf-8')))
    db.commit()
    course_id = course.id
    db.close()

    started = time.time()
    task = asyncio.create_task(build_course(course_id))
    last = ''
    while not task.done():
        await asyncio.sleep(2)
        db = SessionLocal()
        progress = db.query(Course.progress).filter(Course.id == course_id).scalar()
        db.close()
        if progress != last:
            print(f'[{time.time() - started:5.0f} c] {progress}', flush=True)
            last = progress
    await task

    db = SessionLocal()
    course = db.query(Course).filter(Course.id == course_id).first()
    print(f'Статус: {course.status.value}. {course.progress or ""} {course.error or ""}')
    if course.status == CourseStatus.review:
        course.status = CourseStatus.published
        db.commit()
        concepts, requires = course_graph(db, course_id)
        print(f'Курс #{course_id} опубликован. Карта:\n')
        print(mermaid({cid: c.name for cid, c in concepts.items()}, requires))
        print('\nВход: teacher@ustaz.kg / teacher123, student@ustaz.kg / student123')
    db.close()


if __name__ == '__main__':
    asyncio.run(seed())
