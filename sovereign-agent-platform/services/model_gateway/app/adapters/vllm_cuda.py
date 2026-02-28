from app.adapters.openai_compatible import OpenAICompatibleAdapter


class VllmCudaAdapter(OpenAICompatibleAdapter):
    """Stub adapter used when switching to CUDA-hosted vLLM."""
