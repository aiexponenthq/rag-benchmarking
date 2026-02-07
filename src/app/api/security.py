from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.config.settings import get_settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_api_key(api_key_header: str = Security(api_key_header)) -> str | None:
    settings = get_settings()

    # If no API key is configured on the server, allow access (open mode)
    # WARNING: This is for development/POC only.
    if not settings.api_key:
        return None

    if api_key_header == settings.api_key:
        return api_key_header

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Could not validate credentials",
    )
