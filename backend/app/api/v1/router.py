from fastapi import APIRouter

from backend.app.api.v1.endpoints.health import router as health_router

api_router = APIRouter()

api_router.include_router(
    health_router,
    tags=["Health"]
)