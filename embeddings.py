"""
Embeddings: local (na sua máquina) ou via API do Gemini.

Por que os dois:

- **local** (`sentence-transformers`): sem chave, sem cota, roda offline. Mas
  carrega PyTorch, que ocupa ~1 GB de RAM — mais do que o plano gratuito do
  Streamlit Community Cloud oferece.
- **gemini**: some com o PyTorch da imagem e cabe folgado na nuvem, ao custo
  de uma chamada de API por pedaço na indexação e uma por pergunta.

`TIPO_EMBEDDING=auto` (padrão) usa o local se o pacote estiver instalado e cai
para o Gemini se não estiver — é o que faz o mesmo código rodar na sua máquina
e na nuvem sem editar nada.

IMPORTANTE: os dois geram vetores de dimensões diferentes, então cada tipo tem
seu próprio índice (`indice_faiss_local`, `indice_faiss_gemini`). Trocar de
tipo exige reindexar, e o caminho separado evita o erro de dimensão.
"""

from __future__ import annotations

import os

import config  # noqa: F401  (carrega o .env antes de qualquer leitura)

MODELO_LOCAL = os.getenv(
    "MODELO_EMBEDDING", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
MODELO_GEMINI = os.getenv("MODELO_EMBEDDING_GEMINI", "models/gemini-embedding-001")


def _local_disponivel() -> bool:
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return False
    return True


def tipo_embedding() -> str:
    """Retorna 'local' ou 'gemini'."""
    escolha = os.getenv("TIPO_EMBEDDING", "auto").strip().lower()

    if escolha == "local":
        return "local"
    if escolha == "gemini":
        return "gemini"

    return "local" if _local_disponivel() else "gemini"


def descricao() -> str:
    tipo = tipo_embedding()
    return f"local ({MODELO_LOCAL})" if tipo == "local" else f"Gemini ({MODELO_GEMINI})"


def criar_embeddings():
    tipo = tipo_embedding()

    if tipo == "local":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=MODELO_LOCAL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    chave = (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
    if not chave:
        raise RuntimeError(
            "Embedding pelo Gemini exige GOOGLE_API_KEY. Configure a chave ou "
            "instale o sentence-transformers para rodar o embedding local."
        )

    return GoogleGenerativeAIEmbeddings(model=MODELO_GEMINI, google_api_key=chave)


# Mantido para compatibilidade com quem importava a constante antiga.
MODELO_EMBEDDING = MODELO_LOCAL