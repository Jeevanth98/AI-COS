"""
AI-COS Database Connection and Session Management.
"""

import logging
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager

from config.settings import DATABASE_URL
from database.models import Base

logger = logging.getLogger(__name__)

# ── Engine ────────────────────────────────────────────────────────────────────
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},   # required for SQLite + threads
    echo=False,
)

# Enable WAL mode for better concurrent reads (SQLite)
@event.listens_for(engine, "connect")
def _set_sqlite_pragma(dbapi_conn, _):
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def init_db() -> None:
    """Create all tables if they do not yet exist. Idempotent."""
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialised at %s", DATABASE_URL)


@contextmanager
def get_session() -> Session:
    """Yield a database session, committing on success and rolling back on error."""
    session: Session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
