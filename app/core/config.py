from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "ZomaThon Recommendation API"
    APP_ENV: str = "development"
    API_V1_STR: str = "/api/v1"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # DB (Postgres)
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "your_postgres_password"
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_PORT: str = "5432"
    POSTGRES_DB: str = "zomathon_db"
    DATABASE_URL: str = ""

    # Cache (Redis)
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str = ""
    
    # LLM (Gemini primary, Groq fallback)
    GEMINI_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE_PATH: str = "logs/app.log"

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
