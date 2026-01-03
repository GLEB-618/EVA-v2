from datetime import datetime
from typing import Annotated
from sqlalchemy import CheckConstraint, DateTime, Float, Index, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base

intpk = Annotated[int, mapped_column(primary_key=True)]
str16 = Annotated[str, mapped_column(String(16), nullable=False)]
str64 = Annotated[str, mapped_column(String(64), nullable=False)]
str256 = Annotated[str, mapped_column(String(256), nullable=False)]
txt = Annotated[str, mapped_column(Text, nullable=False)]


class MemoryFacts(Base):
    __tablename__ = "memory_facts"

    id: Mapped[intpk]

    tier: Mapped[str16]         # core, extended
    subject: Mapped[str64]      # user / project:eva / device:pc
    predicate: Mapped[str64]    # gpu / cpu / prefers_sqlalchemy_style
    value: Mapped[txt]

    canonical_key: Mapped[str256]

    confidence: Mapped[float] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        onupdate=func.now(),
    )

    __table_args__ = (
        UniqueConstraint("canonical_key", name="uq_memory_facts_canonical_key"),
        CheckConstraint("tier in ('core', 'extended')", name="ck_memory_facts_tier"),
        CheckConstraint("confidence >= 0.0 and confidence <= 1.0", name="ck_memory_facts_confidence"),
        Index("ix_memory_facts_predicate", "predicate"),
        Index("ix_memory_facts_subject", "subject"),
    )


class EpisodicMemory(Base):
    __tablename__ = "episodic_memory"

    id: Mapped[intpk]

    event_type: Mapped[str64] = mapped_column(String(64), nullable=False)  # chat_message / action / error
    summary: Mapped[txt]
    content: Mapped[txt]  # или JSONB

    importance: Mapped[float] = mapped_column(Float, nullable=False, server_default="0.0")

    source_chat_id: Mapped[int] = mapped_column(nullable=True)
    source_message_id: Mapped[int] = mapped_column(nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        # дедуп для Telegram (если используешь):
        UniqueConstraint("source_chat_id", "source_message_id", name="uq_epi_source"),
        CheckConstraint("importance >= 0.0 and importance <= 1.0", name="ck_epi_importance"),
        Index("ix_epi_created_at", "created_at"),
        Index("ix_epi_event_type", "event_type"),
    )


# class Applications(Base):
#     __tablename__ = "applications"

#     id: Mapped[intpk]
#     name: Mapped[stx]
#     path: Mapped[stx]
#     created_at: Mapped[datetime] = mapped_column(
#         DateTime(timezone=True),
#         server_default=func.now(),
#     )