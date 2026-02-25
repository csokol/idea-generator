"""Embedding wrapper — local (sentence-transformers), Google (Generative AI), or OpenAI."""

from __future__ import annotations

DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_DIMENSION = 384

GOOGLE_DEFAULT_MODEL = "gemini-embedding-001"
GOOGLE_DEFAULT_DIMENSION = 768

OPENAI_DEFAULT_MODEL = "text-embedding-3-small"
OPENAI_DEFAULT_DIMENSION = 1536


class LocalEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Batch-encode *texts* and return list of float vectors."""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return [vec.tolist() for vec in embeddings]


class GoogleEmbedder:
    def __init__(
        self,
        model_name: str = GOOGLE_DEFAULT_MODEL,
        dimension: int | None = None,
    ) -> None:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        kwargs: dict = {"model": model_name}
        if dimension is not None:
            kwargs["output_dimensionality"] = dimension
        self._client = GoogleGenerativeAIEmbeddings(**kwargs)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Batch-encode *texts* via Google Generative AI Embeddings."""
        return self._client.embed_documents(texts)


class OpenAIEmbedder:
    def __init__(
        self,
        model_name: str = OPENAI_DEFAULT_MODEL,
        dimension: int | None = None,
    ) -> None:
        from langchain_openai import OpenAIEmbeddings

        kwargs: dict = {"model": model_name}
        if dimension is not None:
            kwargs["dimensions"] = dimension
        self._client = OpenAIEmbeddings(**kwargs)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Batch-encode *texts* via OpenAI Embeddings."""
        return self._client.embed_documents(texts)


# Backward compat alias
Embedder = LocalEmbedder


def build_embedder(
    provider: str = "openai",
    model: str | None = None,
    dimension: int | None = None,
):
    """Factory: return an embedder instance for the given provider."""
    if provider == "local":
        return LocalEmbedder(model_name=model or DEFAULT_MODEL)
    if provider == "google":
        return GoogleEmbedder(
            model_name=model or GOOGLE_DEFAULT_MODEL,
            dimension=dimension,
        )
    if provider == "openai":
        return OpenAIEmbedder(
            model_name=model or OPENAI_DEFAULT_MODEL,
            dimension=dimension,
        )
    raise ValueError(f"Unknown embed provider: {provider!r}")
