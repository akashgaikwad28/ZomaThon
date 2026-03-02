from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.core.config import settings
from app.core.logger import logger

# Base class for SQLAlchemy models
Base = declarative_base()

# We'll use a local SQLite DB for the hackathon demo if Postgres is not available
# But we connect using the DATABASE_URL pattern for industry grade
if not settings.DATABASE_URL:
    # Fallback for easy demo
    SQLALCHEMY_DATABASE_URL = "sqlite:///./zomathon.db"
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
    )
    logger.info("Using SQLite fallback database")
else:
    # Production Postgres
    try:
        engine = create_engine(settings.DATABASE_URL)
        logger.info(f"Connected to PostgreSQL database at {settings.POSTGRES_SERVER}")
    except Exception as e:
        logger.error(f"Failed to connect to Postgres: {e}. Falling back to SQLite.")
        SQLALCHEMY_DATABASE_URL = "sqlite:///./zomathon.db"
        engine = create_engine(
            SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
        )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """Dependency for FastAPI route endpoints."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
