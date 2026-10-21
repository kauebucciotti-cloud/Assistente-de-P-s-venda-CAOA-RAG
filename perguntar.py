"""
Teste rápido pelo terminal, sem subir o Streamlit. Usa a mesma cascata de
provedores do app.

    python perguntar.py "Como é calculado o score da loja?"
"""

from __future__ import annotations

import sys
import time

import config  # noqa: F401  (carrega o .env antes de qualquer leitura)


def main() -> None:
    if len(sys.argv) < 2:
        print('Uso: python perguntar.py "sua pergunta"')
        raise SystemExit(1)

    pergunta = " ".join(sys.argv[1:])

    from langchain_core.output_parsers import StrOutputParser
    from langchain_community.vectorstores import FAISS

    from app import PROMPT, _motivo, formatar_contexto
    from embeddings import criar_embeddings
    from ingest import CAMINHO_INDICE, construir_indice
    from provedor import ESPERA_FALHA_MS, montar_cascata

    embeddings = criar_embeddings()
    banco = (
        FAISS.load_local(
            str(CAMINHO_INDICE), embeddings, allow_dangerous_deserialization=True
        )
        if CAMINHO_INDICE.exists()
        else construir_indice()
    )

    documentos = banco.as_retriever(search_kwargs={"k": 6}).invoke(pergunta)
    entrada = {
        "contexto": formatar_contexto(documentos),
        "pergunta": pergunta,
        "historico": [],
    }

    cascata = montar_cascata()
    if not cascata:
        raise SystemExit("Nenhuma chave configurada no .env.")

    for rotulo, fabrica in cascata:
        cadeia = PROMPT | fabrica() | StrOutputParser()
        inicio = time.monotonic()
        try:
            print(cadeia.invoke(entrada))
            print(f"\n({rotulo} — {time.monotonic() - inicio:.1f}s)")
            break
        except Exception as erro:
            print(f"[{rotulo} falhou: {_motivo(erro)}]", file=sys.stderr)
            time.sleep(ESPERA_FALHA_MS / 1000)
    else:
        raise SystemExit("Nenhum provedor da cascata respondeu.")

    print("\nTrechos usados:")
    for doc in documentos:
        trilha = " > ".join(
            v for k in ("secao", "subsecao") if (v := doc.metadata.get(k))
        )
        print(f"  - {trilha or 'Guia de Onboarding'}")


if __name__ == "__main__":
    main()
