import json
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from ai import llm
from ai.history import SQLChatHistory
from ai.tutor import build_tutor, session_id, sources
from database.db import get_db
from database.models import Concept, UserProfile
from database.schema import MessageSchema
from api.auth import get_current_user
from api.learn import get_enrollment

tutor_router = APIRouter(prefix='/tutor', tags=['Tutor'])


def tutor_input(course_id: int, message: str, user: UserProfile, db: Session) -> tuple[dict, dict]:
    enrollment = get_enrollment(course_id, user, db)
    question = enrollment.open_question
    topics = [name for (name,) in db.query(Concept.name).filter(Concept.course_id == course_id)
              .order_by(Concept.depth, Concept.id)]
    data = {'question': message, 'course_id': course_id, 'course': enrollment.enrollment_course.title,
            'topics': topics, 'open_question': question.text if question else ''}
    config = {**llm.run_config('tutor', student_id=user.id),
              'configurable': {'session_id': session_id(user.id, course_id)}}
    return data, config


@tutor_router.post('/{course_id}/ask/', response_model=dict)
async def ask(course_id: int, data: MessageSchema, db: Session = Depends(get_db),
              current_user: UserProfile = Depends(get_current_user)):
    tutor_data, config = tutor_input(course_id, data.message, current_user, db)
    result = await build_tutor().ainvoke(tutor_data, config=config)
    return {'answer': result['answer'], 'mode': result['mode'], 'sources': sources(result['docs'])}


@tutor_router.post('/{course_id}/stream/')
async def stream(course_id: int, data: MessageSchema, db: Session = Depends(get_db),
                 current_user: UserProfile = Depends(get_current_user)):
    """Server-Sent Events: сначала режим и источники, потом ответ по кусочкам, в конце done."""
    tutor_data, config = tutor_input(course_id, data.message, current_user, db)

    async def events():
        async for chunk in build_tutor().astream(tutor_data, config=config):
            if 'mode' in chunk:
                yield f'event: mode\ndata: {chunk["mode"]}\n\n'
            if 'docs' in chunk:
                yield f'event: sources\ndata: {json.dumps(sources(chunk["docs"]), ensure_ascii=False)}\n\n'
            if 'answer' in chunk:
                yield f'event: token\ndata: {json.dumps(chunk["answer"], ensure_ascii=False)}\n\n'
        yield 'event: done\ndata: {}\n\n'

    return StreamingResponse(events(), media_type='text/event-stream')


@tutor_router.get('/{course_id}/history/', response_model=list)
async def history(course_id: int, db: Session = Depends(get_db),
                  current_user: UserProfile = Depends(get_current_user)):
    get_enrollment(course_id, current_user, db)
    return [{'role': m.type, 'text': m.text} for m in SQLChatHistory(session_id(current_user.id, course_id)).messages]


@tutor_router.delete('/{course_id}/history/', response_model=dict)
async def clear_history(course_id: int, db: Session = Depends(get_db),
                        current_user: UserProfile = Depends(get_current_user)):
    get_enrollment(course_id, current_user, db)
    SQLChatHistory(session_id(current_user.id, course_id)).clear()
    return {'message': 'История диалога с тьютором очищена'}
