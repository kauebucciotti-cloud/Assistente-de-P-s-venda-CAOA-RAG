"""
Carrega o .env de forma previsível.

Duas armadilhas que este módulo resolve:

1. `load_dotenv()` sem argumento procura a partir do diretório de trabalho.
   Rodar o script de outra pasta faz o .env sumir silenciosamente, e tudo se
   comporta como se as chaves não existissem. Aqui o caminho é fixo: o .env
   ao lado deste arquivo.

2. Por padrão o dotenv NÃO sobrescreve variáveis que já existem no ambiente.
   Se o Windows tiver um GOOGLE_API_KEY antigo ou vazio, ele venceria o .env.
   `override=True` inverte isso: o arquivo do projeto manda.

Todo módulo que lê variáveis importa este aqui ANTES de qualquer outro módulo
do projeto.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ARQUIVO_ENV = Path(__file__).resolve().with_name(".env")

load_dotenv(ARQUIVO_ENV, override=True)


def _carregar_secrets_streamlit() -> None:
    """Na nuvem não existe .env: as chaves vêm de Settings → Secrets.

    O Streamlit Cloud já publica os secrets como variáveis de ambiente, mas
    copiá-los aqui garante o mesmo comportamento em qualquer hospedagem que
    use st.secrets, sem o resto do código precisar saber onde está rodando.
    """
    try:
        import streamlit as st

        for chave, valor in st.secrets.items():
            if isinstance(valor, str) and not os.environ.get(chave):
                os.environ[chave] = valor
    except Exception:
        # Sem streamlit, sem secrets.toml ou fora de um app: segue com o .env.
        pass


_carregar_secrets_streamlit()


def diagnostico() -> str:
    """Resumo de onde o .env foi lido — usado pelo modelos.py."""
    if ARQUIVO_ENV.is_file():
        return f".env lido de {ARQUIVO_ENV}"
    if ARQUIVO_ENV.is_dir():
        return f"ATENÇÃO: {ARQUIVO_ENV} é uma PASTA, não um arquivo."
    return f"ATENÇÃO: não existe .env em {ARQUIVO_ENV.parent}"