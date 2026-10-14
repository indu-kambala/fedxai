from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.app.core.config import settings
from backend.app.db.base import Base


def _engine_url() -> str:
    return settings.database_url


engine = create_engine(
    _engine_url(),
    future=True,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, class_=Session)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    from backend.app import models as _models  # noqa: F401
    from backend.app.models import orm as _orm  # noqa: F401

    if settings.database_url.startswith("sqlite"):
        Base.metadata.create_all(bind=engine)
