from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from typing import AsyncGenerator
import os


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://buildable:buildable123@localhost:5432/buildable"
)

# Creates a connection pool that supports async I/O.
# production-ready settings for Supabase connection pooler
# Production-ready settings (works with local Docker and cloud poolers like Supabase/Neon)
engine = create_async_engine(
    DATABASE_URL,
    echo=True,  # Enable SQL query logging for debugging
    future=True,  # Use SQLAlchemy 2.0 style
    pool_pre_ping=True, # Test connections before using them
    pool_size=5, # Number of connections to keep open in the pool
    max_overflow=10, # Additional connections to allow beyond the pool_size, when the pool is full
    pool_recycle=3600, # Recycle connections after 1 hour
    connect_args={
        "statement_cache_size": 0,  # Required for pgbouncer
        "server_settings": {
            "application_name": "lovable-backend",
            "jit": "off"
        }
    }
)

# async_sessionmaker() creates a factory for new async sessions.
# Every time you call AsyncSessionLocal(), you get a new independent database session.
# class_=AsyncSession → ensures it returns async sessions (not sync ones).
# expire_on_commit=False → means objects remain “usable” even after commit.
# If it were True, SQLAlchemy would clear object state after a commit.

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)

class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    # Creates a database session.
    async with AsyncSessionLocal() as session:
        try:
            # "Pauses" the function and hands out the session object to whoever called get_db(). The caller can use this session to interact with the database.
            yield session
            # When the route finishes using the session, Python returns control back to get_db() — continuing after the yield line.
            await session.commit() # Commits any changes made during the session.
        
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close() # Closes the session, releasing any resources it holds.

