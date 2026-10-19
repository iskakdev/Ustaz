from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from database.db import get_db
from database.models import UserProfile, UserRole
from database.schema import UserRegisterSchema, UserLoginSchema, UserOutSchema
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from config import SECRET_KEY, ALGORITHM, ACCESS_TOKEN_LIFETIME
from datetime import timedelta, datetime, timezone
from jose import jwt, JWTError


pwd_context = CryptContext(schemes=['bcrypt'], deprecated='auto')
oauth2_schema = OAuth2PasswordBearer(tokenUrl='/auth/token/')

auth_router = APIRouter(prefix='/auth', tags=['Auth'])


def get_password_hash(password):
    return pwd_context.hash(password)


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(user_id: int):
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_LIFETIME)
    return jwt.encode({'sub': str(user_id), 'exp': expire}, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(token: str = Depends(oauth2_schema), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = int(payload.get('sub'))
    except (JWTError, TypeError, ValueError):
        raise HTTPException(detail='Неверный токен', status_code=401)

    user_db = db.query(UserProfile).filter(UserProfile.id == user_id).first()
    if not user_db:
        raise HTTPException(detail='Неверный токен', status_code=401)
    return user_db


async def get_teacher(current_user: UserProfile = Depends(get_current_user)):
    if current_user.role != UserRole.teacher:
        raise HTTPException(detail='Только для преподавателей', status_code=403)
    return current_user


def login_user(email: str, password: str, db: Session) -> dict:
    user_db = db.query(UserProfile).filter(UserProfile.email == email).first()
    if not user_db or not verify_password(password, user_db.password):
        raise HTTPException(detail='Неверный email или пароль', status_code=401)
    return {'access_token': create_access_token(user_db.id), 'token_type': 'Bearer'}


@auth_router.post('/register/', response_model=dict)
async def register(user: UserRegisterSchema, db: Session = Depends(get_db)):
    if db.query(UserProfile).filter(UserProfile.email == user.email).first():
        raise HTTPException(detail='Пользователь с таким email уже существует', status_code=400)

    user_db = UserProfile(full_name=user.full_name, email=user.email,
                          password=get_password_hash(user.password), role=user.role)
    db.add(user_db)
    db.commit()
    return {'message': 'Вы успешно зарегистрировались!', 'id': user_db.id}


@auth_router.post('/login/', response_model=dict)
async def login(user: UserLoginSchema, db: Session = Depends(get_db)):
    return login_user(user.email, user.password, db)


@auth_router.post('/token/', response_model=dict, include_in_schema=False)
async def token(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """Для кнопки Authorize в Swagger: она шлёт форму, а не JSON."""
    return login_user(form.username, form.password, db)


@auth_router.get('/me/', response_model=UserOutSchema)
async def me(current_user: UserProfile = Depends(get_current_user)):
    return current_user
