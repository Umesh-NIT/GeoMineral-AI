from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


from backend.app.db.models.mineral_occurrence import MineralOccurrence
from backend.app.db.models.study_area import StudyArea