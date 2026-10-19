from datetime import datetime, timezone
from sqlalchemy import event
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.engine import create_engine
from config import DB_URL

engine = create_engine(DB_URL, connect_args={'check_same_thread': False})


@event.listens_for(engine, 'connect')
def enable_foreign_keys(connection, _):
    # Без этого SQLite молча игнорирует ondelete='SET NULL' / 'CASCADE'
    connection.execute('PRAGMA foreign_keys=ON')

SessionLocal = sessionmaker(bind=engine)

Base = declarative_base()


def utcnow() -> datetime:
    """SQLite хранит даты без пояса, поэтому везде работаем с наивным UTC."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


async def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
