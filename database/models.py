from database.db import Base
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import String, Integer, Float, ForeignKey, DateTime, Enum, Text, Boolean, JSON, UniqueConstraint
from typing import Optional, List
from enum import Enum as PyEnum
from datetime import datetime
from sqlalchemy import func


class UserRole(PyEnum):
    student = 'student'
    teacher = 'teacher'


class CourseStatus(PyEnum):
    draft = 'draft'            # преподаватель загружает материалы
    building = 'building'      # ИИ собирает карту понятий и вопросы
    review = 'review'          # карта готова, преподаватель проверяет вопросы
    published = 'published'    # студенты могут учиться
    failed = 'failed'


class QuestionKind(PyEnum):
    recall = 'recall'      # вспомнить определение
    apply = 'apply'        # применить к задаче
    analyze = 'analyze'    # объяснить почему / сравнить / найти ошибку


class Verdict(PyEnum):
    correct = 'correct'
    partial = 'partial'
    wrong = 'wrong'


class UserProfile(Base):
    __tablename__ = 'user'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    full_name: Mapped[str] = mapped_column(String)
    email: Mapped[str] = mapped_column(String, unique=True)
    password: Mapped[str] = mapped_column(String)
    role: Mapped[UserRole] = mapped_column(Enum(UserRole), default=UserRole.student)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    teacher_courses: Mapped[List['Course']] = relationship('Course', back_populates='course_teacher',
                                                           cascade='all, delete-orphan')


class Course(Base):
    __tablename__ = 'course'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String)
    description: Mapped[str] = mapped_column(Text, default='')
    teacher_id: Mapped[int] = mapped_column(ForeignKey('user.id', ondelete='CASCADE'))
    status: Mapped[CourseStatus] = mapped_column(Enum(CourseStatus), default=CourseStatus.draft)
    progress: Mapped[str] = mapped_column(String, default='')
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    course_teacher: Mapped[UserProfile] = relationship(UserProfile, back_populates='teacher_courses')
    course_materials: Mapped[List['Material']] = relationship('Material', back_populates='material_course',
                                                              cascade='all, delete-orphan')
    course_concepts: Mapped[List['Concept']] = relationship('Concept', back_populates='concept_course',
                                                            cascade='all, delete-orphan')
    course_enrollments: Mapped[List['Enrollment']] = relationship('Enrollment', back_populates='enrollment_course',
                                                                  cascade='all, delete-orphan')


class Material(Base):
    __tablename__ = 'material'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    course_id: Mapped[int] = mapped_column(ForeignKey('course.id', ondelete='CASCADE'))
    title: Mapped[str] = mapped_column(String)
    text: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    material_course: Mapped[Course] = relationship(Course, back_populates='course_materials')


class Concept(Base):
    __tablename__ = 'concept'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    course_id: Mapped[int] = mapped_column(ForeignKey('course.id', ondelete='CASCADE'))
    name: Mapped[str] = mapped_column(String)
    summary: Mapped[str] = mapped_column(Text)
    section: Mapped[str] = mapped_column(String, default='')
    misconceptions: Mapped[list] = mapped_column(JSON, default=list)
    depth: Mapped[int] = mapped_column(Integer, default=0)

    concept_course: Mapped[Course] = relationship(Course, back_populates='course_concepts')
    concept_questions: Mapped[List['Question']] = relationship('Question', back_populates='question_concept',
                                                               cascade='all, delete-orphan')
    requires: Mapped[List['ConceptLink']] = relationship('ConceptLink', foreign_keys='ConceptLink.concept_id',
                                                         back_populates='link_concept', cascade='all, delete-orphan')


class ConceptLink(Base):
    """Ребро карты: чтобы понять concept, нужно сначала понять requires."""
    __tablename__ = 'concept_link'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    concept_id: Mapped[int] = mapped_column(ForeignKey('concept.id', ondelete='CASCADE'))
    requires_id: Mapped[int] = mapped_column(ForeignKey('concept.id', ondelete='CASCADE'))

    link_concept: Mapped[Concept] = relationship(Concept, foreign_keys=[concept_id], back_populates='requires')


class Question(Base):
    __tablename__ = 'question'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    concept_id: Mapped[int] = mapped_column(ForeignKey('concept.id', ondelete='CASCADE'))
    kind: Mapped[QuestionKind] = mapped_column(Enum(QuestionKind))
    text: Mapped[str] = mapped_column(Text)
    reference_answer: Mapped[str] = mapped_column(Text)
    key_points: Mapped[list] = mapped_column(JSON, default=list)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    question_concept: Mapped[Concept] = relationship(Concept, back_populates='concept_questions')


class Enrollment(Base):
    __tablename__ = 'enrollment'
    __table_args__ = (UniqueConstraint('student_id', 'course_id'),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    student_id: Mapped[int] = mapped_column(ForeignKey('user.id', ondelete='CASCADE'))
    course_id: Mapped[int] = mapped_column(ForeignKey('course.id', ondelete='CASCADE'))
    open_question_id: Mapped[Optional[int]] = mapped_column(ForeignKey('question.id', ondelete='SET NULL'),
                                                            nullable=True)
    open_mode: Mapped[str] = mapped_column(String, default='')
    diagnostic_left: Mapped[int] = mapped_column(Integer, default=3)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    enrollment_course: Mapped[Course] = relationship(Course, back_populates='course_enrollments')
    open_question: Mapped[Optional[Question]] = relationship(Question)


class Mastery(Base):
    """Что студент знает о понятии: вероятность знания (BKT) и устойчивость памяти в днях."""
    __tablename__ = 'mastery'
    __table_args__ = (UniqueConstraint('student_id', 'concept_id'),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    student_id: Mapped[int] = mapped_column(ForeignKey('user.id', ondelete='CASCADE'))
    concept_id: Mapped[int] = mapped_column(ForeignKey('concept.id', ondelete='CASCADE'))
    p_known: Mapped[float] = mapped_column(Float)
    stability: Mapped[float] = mapped_column(Float, default=1.0)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    last_seen_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    last_question_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)


class Attempt(Base):
    __tablename__ = 'attempt'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    student_id: Mapped[int] = mapped_column(ForeignKey('user.id', ondelete='CASCADE'))
    question_id: Mapped[int] = mapped_column(ForeignKey('question.id', ondelete='CASCADE'))
    mode: Mapped[str] = mapped_column(String)
    answer: Mapped[str] = mapped_column(Text)
    score: Mapped[float] = mapped_column(Float)
    verdict: Mapped[Verdict] = mapped_column(Enum(Verdict))
    feedback: Mapped[str] = mapped_column(Text)
    misconception: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    graded_by: Mapped[str] = mapped_column(String, default='llm')
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    attempt_question: Mapped[Question] = relationship(Question)


class ChatMessage(Base):
    """История диалога с тьютором, её читает и пишет SQLChatHistory."""
    __tablename__ = 'chat_message'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String, index=True)
    message: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class LLMCache(Base):
    __tablename__ = 'llm_cache'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(String, unique=True)
    generations: Mapped[list] = mapped_column(JSON)
    hits: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class AIUsage(Base):
    __tablename__ = 'ai_usage'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    feature: Mapped[str] = mapped_column(String, index=True)
    model: Mapped[str] = mapped_column(String)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    cached: Mapped[bool] = mapped_column(Boolean, default=False)
    ok: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
