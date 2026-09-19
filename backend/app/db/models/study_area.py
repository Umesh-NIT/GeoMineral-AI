from datetime import datetime

from geoalchemy2 import Geometry
from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.db.base import Base


class StudyArea(Base):
    __tablename__ = "study_areas"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    crs: Mapped[str] = mapped_column(String(50), nullable=False, default="EPSG:4326")
    geometry: Mapped[object] = mapped_column(
        Geometry("POLYGON", srid=4326),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )