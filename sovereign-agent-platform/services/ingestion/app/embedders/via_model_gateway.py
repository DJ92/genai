import httpx


class ModelGatewayEmbedder:
    def __init__(self, model_gateway_url: str) -> None:
        self._url = model_gateway_url.rstrip("/")

    async def embed(self, text: str) -> list[float]:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(f"{self._url}/v1/embeddings", json={"input": text})
            response.raise_for_status()
            payload = response.json()
            data = payload.get("data", [])
            if not data:
                return []
            return data[0].get("embedding", [])
