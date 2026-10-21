"""
Lista os modelos de chat disponíveis em cada provedor configurado.

    python modelos.py

Os nomes de modelo mudam com frequência — modelos são descontinuados sem
aviso. Quando a cascata acusar "modelo indisponível", comece por aqui e ajuste
MODELO_GEMINI / MODELO_GROQ / MODELO_OPENAI / MODELO_XAI no .env.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import config  # noqa: E402

from provedor import BASE_XAI, ORDEM, ROTULOS, _chave  # noqa: E402

# A OpenAI e a xAI respondem no formato {"data": [{"id": ...}]}; o Google
# responde {"models": [{"name": "models/..."}]}.
FONTES = {
    "openai": ("https://api.openai.com/v1/models", "bearer"),
    "groq": ("https://api.groq.com/openai/v1/models", "bearer"),
    "xai": (f"{BASE_XAI}/models", "bearer"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/models", "query"),
}

RUIDO = (
    "whisper",
    "tts",
    "embedding",
    "image",
    "guard",
    "moderation",
    "audio",
    "transcribe",
    "lyria",
    "dall-e",
    "realtime",
)


def listar(provedor: str, chave: str) -> list[str]:
    url, autenticacao = FONTES[provedor]

    if autenticacao == "query":
        requisicao = urllib.request.Request(f"{url}?key={chave}")
    else:
        requisicao = urllib.request.Request(
            url, headers={"Authorization": f"Bearer {chave}"}
        )

    with urllib.request.urlopen(requisicao, timeout=30) as resposta:
        dados = json.load(resposta)

    if provedor == "gemini":
        nomes = [
            m["name"].removeprefix("models/")
            for m in dados.get("models", [])
            if "generateContent" in m.get("supportedGenerationMethods", [])
        ]
    else:
        nomes = [m["id"] for m in dados.get("data", [])]

    return sorted(n for n in nomes if not any(r in n.lower() for r in RUIDO))


def main() -> None:
    print(config.diagnostico())
    algum = False

    for provedor in ORDEM:
        chave = _chave(provedor)
        if not chave:
            continue

        algum = True
        rotulo = ROTULOS.get(provedor, provedor)
        print(f"\n=== {rotulo} ===")
        try:
            for modelo in listar(provedor, chave):
                print(f"  {modelo}")
        except urllib.error.HTTPError as erro:
            print(f"  erro {erro.code}: {erro.reason} (chave inválida ou sem acesso)")
        except Exception as erro:  # rede, DNS, proxy
            print(f"  falhou: {erro}")

    if not algum:
        raise SystemExit(
            "Nenhuma chave no .env. Preencha GOOGLE_API_KEY, GROQ_API_KEY, "
            "OPENAI_API_KEY ou XAI_API_KEY."
        )


if __name__ == "__main__":
    main()
