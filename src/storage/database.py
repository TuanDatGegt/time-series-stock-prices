from __future__ import annotations

from sqlalchemy import Column, Float, Integer, String, TIMESTAMP, UniqueConstraint, create_engine as sqlalchemy_create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class MarketData(Base):
    __tablename__ = "market_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False)
    timestamp = Column(TIMESTAMP, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False)

    __table_args__ = (
        UniqueConstraint("symbol", "timestamp", name="uq_symbol_timestamp"),
        {"sqlite_autoincrement": True},
    )


def create_engine(connection_url: str) -> Engine:
    return sqlalchemy_create_engine(connection_url)


def init_db(engine: Engine | None = None) -> Engine:
    if engine is None:
        engine = create_engine("sqlite:///:memory:")

    Base.metadata.create_all(bind=engine)
    return engine


SessionLocal = sessionmaker(autocommit=False, autoflush=False)
