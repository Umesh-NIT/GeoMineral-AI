from datetime import datetime

from geoalchemy2 import Geometry
from sqlalchemy import DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.db.base import Base


class MineralOccurrence(Base):
    __tablename__ = "mineral_occurrences"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    source: Mapped[str] = mapped_column(String(100), nullable=False)
    occurrence_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    mineral: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    commodity: Mapped[str | None] = mapped_column(String(100), nullable=True)
    deposit_type: Mapped[str | None] = mapped_column(String(255), nullable=True)
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    geometry: Mapped[object] = mapped_column(
        Geometry("POINT", srid=4326),
        nullable=False,
    )
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    source_reference: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )