"""
Indexação: documento -> pedaços -> embeddings -> banco vetorial FAISS.

Rode uma vez (ou sempre que o documento mudar):

    python ingest.py

O índice fica salvo em ./indice_faiss, então o app sobe rápido e você não
paga embedding de novo a cada execução.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

import config  # noqa: F401  (carrega o .env antes de qualquer leitura)
from embeddings import criar_embeddings, descricao, tipo_embedding
from loader import docx_para_markdown

CAMINHO_DOCUMENTO = Path(
    os.getenv("CAMINHO_DOCUMENTO", "data/Guia_Onboarding_Pos-venda.docx")
)
# Um índice por tipo de embedding: os vetores têm dimensões diferentes, e
# misturar os dois dá erro de shape na primeira busca.
CAMINHO_INDICE = Path(
    os.getenv("CAMINHO_INDICE", f"indice_faiss_{tipo_embedding()}")
)

TAMANHO_PEDACO = 1200
SOBREPOSICAO = 200


def _reparar_tabelas(pedacos: list[Document]) -> list[Document]:
    """Repõe o cabeçalho em tabelas longas que foram cortadas entre pedaços.

    Tabelas grandes (3.1 e o glossário) não caber em um pedaço só. Sem isso, a
    continuação chega ao modelo como linhas soltas de "| x | y |", sem dizer o
    que é cada coluna.
    """
    cabecalho_atual: str | None = None

    for pedaco in pedacos:
        linhas = pedaco.page_content.splitlines()
        tem_cabecalho = any(set(l.replace("|", "").strip()) <= {"-", " "} and "|" in l for l in linhas)

        if tem_cabecalho:
            for i, linha in enumerate(linhas):
                if set(linha.replace("|", "").strip()) <= {"-", " "} and "|" in linha and i > 0:
                    cabecalho_atual = linhas[i - 1] + "\n" + linha
                    break
        elif cabecalho_atual and any(l.startswith("|") for l in linhas):
            pedaco.page_content = cabecalho_atual + "\n" + pedaco.page_content

    return pedacos


def quebrar_em_pedacos(markdown: str) -> list[Document]:
    """Quebra primeiro por seção (títulos) e depois por tamanho.

    A quebra por título garante que cada pedaço carregue na metadata a seção
    de onde veio ("3. Os indicadores" > "3.2 Faróis"). Isso serve para dois
    fins: dar contexto ao modelo e permitir citar a fonte na resposta.
    """
    por_secao = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "secao"), ("##", "subsecao")],
        strip_headers=False,
    ).split_text(markdown)

    por_tamanho = RecursiveCharacterTextSplitter(
        chunk_size=TAMANHO_PEDACO,
        chunk_overlap=SOBREPOSICAO,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    pedacos = _reparar_tabelas(por_tamanho.split_documents(por_secao))

    # Prefixa cada pedaço com a sua trilha de seção: ajuda o retriever a
    # achar o pedaço certo quando a pergunta usa o nome da seção.
    for pedaco in pedacos:
        trilha = " > ".join(
            v for k in ("secao", "subsecao") if (v := pedaco.metadata.get(k))
        )
        if trilha and not pedaco.page_content.startswith(trilha):
            pedaco.page_content = f"[{trilha}]\n{pedaco.page_content}"

    return pedacos


def construir_indice(forcar: bool = False) -> FAISS:
    # Sem chave aqui: o embedding roda local. Na primeira execução o modelo
    # é baixado para o cache do Hugging Face.
    print(f"Embedding: {descricao()}")
    embeddings = criar_embeddings()

    if CAMINHO_INDICE.exists() and not forcar:
        print(f"Índice já existe em {CAMINHO_INDICE}. Carregando.")
        return FAISS.load_local(
            str(CAMINHO_INDICE), embeddings, allow_dangerous_deserialization=True
        )

    if not CAMINHO_DOCUMENTO.exists():
        raise FileNotFoundError(f"Documento não encontrado: {CAMINHO_DOCUMENTO}")

    print(f"Lendo {CAMINHO_DOCUMENTO}")
    markdown = docx_para_markdown(CAMINHO_DOCUMENTO)

    pedacos = quebrar_em_pedacos(markdown)
    print(f"{len(pedacos)} pedaços gerados. Gerando embeddings...")

    banco = FAISS.from_documents(pedacos, embeddings)

    if CAMINHO_INDICE.exists():
        shutil.rmtree(CAMINHO_INDICE)
    banco.save_local(str(CAMINHO_INDICE))
    print(f"Índice salvo em {CAMINHO_INDICE}")
    return banco


if __name__ == "__main__":
    import sys

    construir_indice(forcar="--forcar" in sys.argv)