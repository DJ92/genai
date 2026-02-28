from app.adapters.openai_compatible import OpenAICompatibleAdapter


class LlamaCppLocalAdapter(OpenAICompatibleAdapter):
    """Thin alias for local llama.cpp-specific behavior when needed."""
