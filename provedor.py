"""
Cascata de provedores: Gemini (Google), Groq, ChatGPT (OpenAI) e Grok (xAI).

Gemini e Groq têm camada gratuita; OpenAI e xAI cobram por uso (a assinatura
do ChatGPT ou do Grok no app não dá crédito de API — são produtos separados).
Por isso a ordem padrão começa pelos gratuitos.

A ideia é não depender de nenhum fornecedor: o app tenta o primeiro da ordem e,
se ele falhar — cota, 429, 503, modelo descontinuado, timeout —, passa para o
próximo em ~200 ms. Só entram na cascata os provedores cuja chave está no .env,
então dá para rodar com um, dois ou três.

Sobre tempo: a troca entre provedores é de milissegundos, mas a resposta de um
LLM leva segundos. O que controla quanto esperamos antes de desistir de um
provedor é o TIMEOUT_SEGUNDOS — baixo demais derruba respostas boas, alto
demais faz você esperar por um serviço que já está morto. 20s é um meio-termo
razoável para respostas curtas como as deste assistente.
"""

from __future__ import annotations

import os

# A xAI e a OpenAI falam o mesmo protocolo; muda só a URL base.
BASE_XAI = "https://api.x.ai/v1"

TIMEOUT_SEGUNDOS = float(os.getenv("TIMEOUT_SEGUNDOS", "20"))
ESPERA_FALHA_MS = int(os.getenv("ESPERA_FALHA_MS", "200"))

# Ordem de preferência. Mude em ORDEM_PROVEDORES no .env.
ORDEM = [
    p.strip().lower()
    for p in os.getenv("ORDEM_PROVEDORES", "gemini,groq,openai,xai").split(",")
    if p.strip()
]

# Um ou mais modelos por provedor, separados por vírgula. Os `-latest` do
# Gemini sobrevivem à troca de versões; os outros dois precisam de nome fixo,
# por isso o modelos.py existe.
MODELOS = {
    "gemini": os.getenv("MODELO_GEMINI", "gemini-flash-latest,gemini-3.8-flash"),
    "groq": os.getenv(
        "MODELO_GROQ", "llama-3.3-70b-versatile,openai/gpt-oss-120b,llama-3.1-8b-instant"
    ),
    "openai": os.getenv("MODELO_OPENAI", "gpt-5-mini,gpt-4.1-mini"),
    "xai": os.getenv("MODELO_XAI", "grok-4-fast,grok-3-mini"),
}

ROTULOS = {
    "gemini": "Gemini",
    "groq": "Groq",
    "openai": "ChatGPT",
    "xai": "Grok",
}

# Quais exigem cartão. Só serve para avisar na interface.
PAGOS = {"openai", "xai"}

# Chaves aceitas por provedor. GROQ_API_KEY continua sendo lida para quem já
# tinha o .env antigo: se ela contiver uma chave `xai-`, vale como xAI.
def _chave(provedor: str) -> str:
    if provedor == "openai":
        return (os.getenv("OPENAI_API_KEY") or "").strip()

    if provedor == "groq":
        chave = (os.getenv("GROQ_API_KEY") or "").strip()
        # Um `xai-` em GROQ_API_KEY é chave da xAI guardada no lugar errado:
        # não vale como Groq.
        return chave if chave.startswith("gsk_") else ""

    if provedor == "xai":
        direta = (os.getenv("XAI_API_KEY") or "").strip()
        if direta:
            return direta
        herdada = (os.getenv("GROQ_API_KEY") or "").strip()
        return herdada if herdada.startswith("xai-") else ""

    if provedor == "gemini":
        return (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()

    return ""


def _criar_openai(modelo: str, chave: str, base_url: str | None = None):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=modelo,
        temperature=0.1,
        api_key=chave,
        base_url=base_url,
        timeout=TIMEOUT_SEGUNDOS,
        max_retries=0,  # o retry é nosso, para poder trocar de provedor
    )


def _criar_groq(modelo: str, chave: str):
    from langchain_groq import ChatGroq

    return ChatGroq(
        model=modelo,
        temperature=0.1,
        api_key=chave,
        timeout=TIMEOUT_SEGUNDOS,
        max_retries=0,
    )


def _criar_gemini(modelo: str, chave: str):
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=modelo,
        temperature=0.1,
        google_api_key=chave,
        timeout=TIMEOUT_SEGUNDOS,
        max_retries=0,
    )


def montar_cascata() -> list[tuple[str, callable]]:
    """Monta a lista [(rótulo, fábrica_do_cliente)] na ordem configurada.

    Devolve fábricas e não clientes prontos para que nada seja instanciado —
    nem valide chave — antes de a pergunta existir.
    """
    cascata: list[tuple[str, callable]] = []

    for provedor in ORDEM:
        chave = _chave(provedor)
        if not chave:
            continue

        for modelo in (m.strip() for m in MODELOS.get(provedor, "").split(",")):
            if not modelo:
                continue

            rotulo = f"{ROTULOS.get(provedor, provedor)} · {modelo}"

            if provedor == "openai":
                cascata.append((rotulo, lambda m=modelo, k=chave: _criar_openai(m, k)))
            elif provedor == "groq":
                cascata.append((rotulo, lambda m=modelo, k=chave: _criar_groq(m, k)))
            elif provedor == "xai":
                cascata.append(
                    (rotulo, lambda m=modelo, k=chave: _criar_openai(m, k, BASE_XAI))
                )
            elif provedor == "gemini":
                cascata.append((rotulo, lambda m=modelo, k=chave: _criar_gemini(m, k)))

    return cascata


def provedores_ativos() -> list[str]:
    """Quais provedores têm chave configurada, na ordem de preferência."""
    return [ROTULOS.get(p, p) for p in ORDEM if _chave(p)]
