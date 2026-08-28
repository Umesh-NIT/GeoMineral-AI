from fastapi import APIRouter
from sqlalchemy import text

from backend.app.db.session import engine


router = APIRouter()


@router.get("/health")
def health():
    return {
        "status": "healthy"
    }


@router.get("/health/database")
def database_health():
    with engine.connect() as connection:
        result = connection.execute(text("SELECT 1"))
        value = result.scalar()

    return {
        "database": "connected",
        "result": value
    }