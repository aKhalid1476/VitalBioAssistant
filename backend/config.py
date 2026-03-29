from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # API keys
    anthropic_api_key: str = Field(..., alias="ANTHROPIC_API_KEY")
    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")

    # ChromaDB
    chroma_persist_dir: str = Field("./chroma_db", alias="CHROMA_PERSIST_DIR")
    chroma_collection_name: str = Field("bloodborne_pathogens", alias="CHROMA_COLLECTION_NAME")

    # Ingestion
    pdf_path: str = Field("./data/pathogenInfo.pdf", alias="PDF_PATH")
    chunk_size: int = Field(800, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(150, alias="CHUNK_OVERLAP")

    # Retrieval
    retriever_k: int = Field(5, alias="RETRIEVER_K")

    # LLM
    llm_model: str = Field("claude-sonnet-4-6", alias="LLM_MODEL")
    llm_temperature: float = Field(0.0, alias="LLM_TEMPERATURE")

    # Embeddings
    embedding_model: str = Field("text-embedding-3-small", alias="EMBEDDING_MODEL")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "populate_by_name": True}


settings = Settings()
