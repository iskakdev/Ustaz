from fastapi import FastAPI
from langchain_core.globals import set_llm_cache
from api import auth, course, learn, tutor, coach, insight
from ai.cache import SQLCache
from database.db import Base, engine
import uvicorn

Base.metadata.create_all(bind=engine)
set_llm_cache(SQLCache())

ustaz_app = FastAPI(title='Ustaz', description='Курс как карта понятий: ИИ знает, что ты уже понял')
ustaz_app.include_router(auth.auth_router)
ustaz_app.include_router(course.course_router)
ustaz_app.include_router(learn.learn_router)
ustaz_app.include_router(tutor.tutor_router)
ustaz_app.include_router(coach.coach_router)
ustaz_app.include_router(insight.insight_router)

if __name__ == '__main__':
    uvicorn.run(ustaz_app, host='127.0.0.1', port=8001)
