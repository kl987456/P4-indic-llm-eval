from __future__ import annotations
import uuid
from datetime import datetime
from sqlalchemy import DateTime, Float, JSON, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column
from api.db.database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


class EvalRun(Base):
    __tablename__ = "eval_runs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    task_names: Mapped[list] = mapped_column(JSON, default=list)
    model_backend: Mapped[str] = mapped_column(String(64), nullable=False)
    model_name: Mapped[str] = mapped_column(String(256), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="pending")
    results_json: Mapped[str | None] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime)
