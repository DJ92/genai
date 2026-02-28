import httpx

from app.core.config import get_settings


class OpenAICompatibleAdapter:
    def __init__(self) -> None:
        self._settings = get_settings()

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self._settings.model_backend_api_key:
            headers["Authorization"] = f"Bearer {self._settings.model_backend_api_key}"
        return headers

    async def chat(self, payload: dict) -> dict:
        url = f"{self._settings.model_backend_base_url}/v1/chat/completions"
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, headers=self._headers(), json=payload)
            response.raise_for_status()
            return response.json()

    async def embeddings(self, payload: dict) -> dict:
        url = f"{self._settings.model_backend_base_url}/v1/embeddings"
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, headers=self._headers(), json=payload)
            response.raise_for_status()
            return response.json()
