from pydantic import BaseModel, EmailStr, Field
from database.models import UserRole, CourseStatus, QuestionKind
from datetime import datetime
from typing import Optional


class UserRegisterSchema(BaseModel):
    full_name: str
    email: EmailStr
    password: str = Field(min_length=6)
    role: UserRole = UserRole.student


class UserLoginSchema(BaseModel):
    email: EmailStr
    password: str


class UserOutSchema(BaseModel):
    id: int
    full_name: str
    email: EmailStr
    role: UserRole
    created_at: datetime


class CourseInputSchema(BaseModel):
    title: str
    description: str = ''


class CourseOutSchema(BaseModel):
    id: int
    title: str
    description: str
    teacher_id: int
    status: CourseStatus
    progress: str
    error: Optional[str]
    created_at: datetime


class MaterialInputSchema(BaseModel):
    title: str
    text: str = Field(min_length=50)


class MaterialOutSchema(BaseModel):
    id: int
    course_id: int
    title: str
    created_at: datetime


class QuestionOutSchema(BaseModel):
    id: int
    concept_id: int
    kind: QuestionKind
    text: str
    reference_answer: str
    key_points: list[str]
    is_active: bool


class QuestionInputSchema(BaseModel):
    kind: QuestionKind
    text: str
    reference_answer: str
    key_points: list[str] = Field(min_length=1)


class QuestionUpdateSchema(BaseModel):
    text: Optional[str] = None
    reference_answer: Optional[str] = None
    key_points: Optional[list[str]] = None
    is_active: Optional[bool] = None


class ConceptUpdateSchema(BaseModel):
    name: Optional[str] = None
    summary: Optional[str] = None
    misconceptions: Optional[list[str]] = None
    requires: Optional[list[int]] = None


class AnswerInputSchema(BaseModel):
    question_id: int
    answer: str


class MessageSchema(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
