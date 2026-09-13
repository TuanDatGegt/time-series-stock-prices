## src/storage/database.py

"""
Module: src/storage/database.py
Description: Database Engine and Session Initialization.
How it works:
    Provides utility functions to instantiate SQLAlchemy database engines and sessions.
    Configures session factory (`SessionLocal`) and initializes database tables based on
    declarative ORM metadata (`Base.metadata.create_all`).
"""

from __future__ import annotations
from sqlalchemy import create_engine as sqlalchemy_create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Declarative base class for SQLAlchemy ORM models
Base = declarative_base()


def create_engine(connection_url: str) -> Engine:
    """
    Create a SQLAlchemy Engine instance from a database connection URL.

    Args:
        connection_url (str): Database connection string (e.g., 'sqlite:///:memory:' or PostgreSQL URL).

    Returns:
        Engine: Initialized SQLAlchemy Engine.
    """
    return sqlalchemy_create_engine(connection_url)


def init_db(engine: Engine | None = None) -> Engine:
    """
    Initialize the database by creating all defined ORM tables.

    Args:
        engine (Engine | None): Optional target database engine. Defaults to in-memory SQLite if None.

    Returns:
        Engine: The database engine bound to created tables.
    """
    if engine is None:
        engine = create_engine("sqlite:///:memory:")

    Base.metadata.create_all(bind=engine)
    return engine


# Session factory for managing database transaction sessions
SessionLocal = sessionmaker(autocommit=False, autoflush=False)
