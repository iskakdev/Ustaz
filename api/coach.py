from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from ai.coach import CoachContext, ask_coach
from database.db import get_db
from database.models import UserProfile
from database.schema import MessageSchema
from api.auth import get_current_user
from api.learn import get_enrollment

coach_router = APIRouter(prefix='/coach', tags=['Coach'])


@coach_router.post('/{course_id}/chat/', response_model=dict)
async def coach_chat(course_id: int, data: MessageSchema, db: Session = Depends(get_db),
                     current_user: UserProfile = Depends(get_current_user)):
    enrollment = get_enrollment(course_id, current_user, db)
    context = CoachContext(student_id=current_user.id, course_id=course_id, student_name=current_user.full_name,
                           course_title=enrollment.enrollment_course.title)
    return await ask_coach(context, data.message)
