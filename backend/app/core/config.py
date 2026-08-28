from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "GeoMineral AI"
    app_env: str = "development"
    debug: bool = True

    frontend_url: str = "http://localhost:5173"

    database_url: str = "postgresql://postgres:postgres@localhost:5432/geomineral"

    model_path: str = "models/trained/prospectivity_model.joblib"

    llm_api_key: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()