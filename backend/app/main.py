from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.v1.router import api_router
from backend.app.core.config import settings


app = FastAPI(
    title=settings.app_name,
    description="AI-powered mineral prospectivity mapping API",
    version="0.1.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        settings.frontend_url,
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(
    api_router,
    prefix="/api/v1"
)


@app.get("/")
def root():
    return {
        "project": settings.app_name,
        "status": "running",
        "version": "0.1.0"
    }