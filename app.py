"""
Assistente de Pós-venda — chat RAG sobre o Guia de Onboarding.

    streamlit run app.py
"""

from __future__ import annotations

import os

import time

import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_community.vectorstores import FAISS

import config  # noqa: F401  (carrega o .env antes de qualquer leitura)
from embeddings import criar_embeddings  # noqa: E402
from ingest import CAMINHO_INDICE, construir_indice  # noqa: E402
from provedor import (  # noqa: E402
    ESPERA_FALHA_MS,
    montar_cascata,
    provedores_ativos,
)

PEDACOS_RECUPERADOS = 6

SYSTEM = """\
Você é o assistente de Pós-venda da CAOA (redes D21/Chery e A21/Changan).
Seu público são Aprendizes, Estagiários e Analistas que estão chegando à área.

Regras:
- Responda SOMENTE com base no CONTEXTO abaixo, que vem do Guia de Onboarding
  de Pós-venda. Não use conhecimento geral sobre o setor automotivo.
- Se a resposta não estiver no contexto, diga claramente que o guia não cobre
  isso e sugira a quem perguntar (gestor regional ou gerente corporativo).
- Números, pesos, faixas de farol e nomes de sistemas devem ser citados
  exatamente como estão no contexto. Nunca arredonde nem estime.
- Seja direto: resposta primeiro, detalhe depois. Use tabela quando o contexto
  trouxer tabela.
- Ao final, cite entre parênteses a seção do guia que embasou a resposta.

CONTEXTO:
{contexto}"""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM),
        ("placeholder", "{historico}"),
        ("human", "{pergunta}"),
    ]
)


# Identificamos o tipo de falha pelo conteúdo da mensagem, não pela classe da
# exceção: cada integração reembrulha o erro do SDK numa classe própria, então
# capturar a exceção original não funciona de forma confiável.
SINAIS_SOBRECARGA = (
    "503",
    "429",
    "UNAVAILABLE",
    "high demand",
    "overloaded",
    "rate limit",
    "Rate limit",
)
SINAIS_INDISPONIVEL = (
    "404",
    "NOT_FOUND",
    "not found",
    "no longer available",
    "decommissioned",
    "does not exist",
)

SINAIS_CHAVE = (
    "401",
    "403",
    "invalid_api_key",
    "Incorrect API key",
    "API key not valid",
    "PERMISSION_DENIED",
    "insufficient_quota",
)


def _contem(erro: Exception, sinais: tuple[str, ...]) -> bool:
    return any(sinal in str(erro) for sinal in sinais)


def _e_sobrecarga(erro: Exception) -> bool:
    return _contem(erro, SINAIS_SOBRECARGA)


def _e_indisponivel(erro: Exception) -> bool:
    return _contem(erro, SINAIS_INDISPONIVEL)


def _e_chave(erro: Exception) -> bool:
    return _contem(erro, SINAIS_CHAVE)


def _motivo(erro: Exception) -> str:
    if _e_sobrecarga(erro):
        return "sobrecarregado"
    if _e_indisponivel(erro):
        return "modelo indisponível"
    if _e_chave(erro):
        return "chave rejeitada"
    if "timeout" in str(erro).lower() or "timed out" in str(erro).lower():
        return "tempo esgotado"
    return type(erro).__name__


def responder_com_cascata(cadeias, entrada, caixa) -> tuple[str, list[str]]:
    """Percorre a cascata até um provedor responder.

    Retorna (texto, trilha), onde a trilha registra quem falhou e por quê até
    alguém responder. Não há retry no mesmo provedor: se ele falhou, o próximo
    tem muito mais chance de responder rápido do que o mesmo tentando de novo.
    A troca leva ESPERA_FALHA_MS (200 ms por padrão) — o suficiente para não
    martelar as APIs em rajada, e imperceptível para quem está esperando.

    O texto parcial de uma tentativa que falhou é descartado: a saída vai para
    um container que é limpo antes da próxima.
    """
    trilha: list[str] = []
    ultimo_erro: Exception | None = None

    for indice, (rotulo, cadeia) in enumerate(cadeias):
        try:
            with caixa.container():
                if trilha:
                    st.caption(f"Tentando {rotulo}...")
                texto = st.write_stream(cadeia.stream(entrada))
            return texto, trilha + [f"✅ {rotulo}"]

        except Exception as erro:
            ultimo_erro = erro
            trilha.append(f"❌ {rotulo} ({_motivo(erro)})")

            if indice == len(cadeias) - 1:
                raise

            time.sleep(ESPERA_FALHA_MS / 1000)

    raise ultimo_erro or RuntimeError("Nenhum provedor configurado.")


def formatar_contexto(documentos) -> str:
    blocos = []
    for i, doc in enumerate(documentos, start=1):
        blocos.append(f"--- Trecho {i} ---\n{doc.page_content}")
    return "\n\n".join(blocos)


@st.cache_resource(show_spinner="Carregando o índice do documento...")
def carregar_cadeia():
    """Monta o retriever e a cadeia LCEL uma única vez por sessão."""
    embeddings = criar_embeddings()

    if CAMINHO_INDICE.exists():
        banco = FAISS.load_local(
            str(CAMINHO_INDICE), embeddings, allow_dangerous_deserialization=True
        )
    else:
        banco = construir_indice()

    retriever = banco.as_retriever(search_kwargs={"k": PEDACOS_RECUPERADOS})

    # Recupera os trechos e, em paralelo, repassa a pergunta e o histórico.
    cadeia = (
        RunnableParallel(
            documentos=RunnableLambda(lambda x: x["pergunta"]) | retriever,
            pergunta=RunnablePassthrough(),
            historico=RunnablePassthrough(),
        )
        | RunnableLambda(
            lambda x: {
                "contexto": formatar_contexto(x["documentos"]),
                "pergunta": x["pergunta"]["pergunta"],
                "historico": x["historico"]["historico"],
                "documentos": x["documentos"],
            }
        )
    )

    # Uma cadeia de resposta por entrada da cascata. São objetos leves: nada
    # sai pela rede até o .stream() ser chamado.
    cadeias_resposta = [
        (rotulo, PROMPT | fabrica() | StrOutputParser())
        for rotulo, fabrica in montar_cascata()
    ]

    return cadeia, cadeias_resposta


SUGESTOES = [
    "Como é calculado o score da loja no Fechamento Regional?",
    "Quais são as faixas de farol do indicador de agendamento?",
    "Onde eu encontro a base de OS em aberto acima de 30 dias?",
    "Qual a diferença entre NPS e CCSI?",
    "Qual a sequência obrigatória da trilha de diagnóstico?",
]


def _liberado() -> bool:
    """Trava opcional por senha, para quando o app está publicado na web.

    Sem SENHA_APP definida, não aparece nada — é o modo local. Com ela, cada
    visitante precisa digitar a senha antes de gastar a sua cota de API, o
    que importa num link público que qualquer um pode abrir.
    """
    senha = (os.getenv("SENHA_APP") or "").strip()
    if not senha:
        return True

    if st.session_state.get("liberado"):
        return True

    digitada = st.text_input("Senha de acesso", type="password")
    if not digitada:
        st.caption("Peça a senha a quem compartilhou o link.")
        return False

    if digitada == senha:
        st.session_state.liberado = True
        st.rerun()

    st.error("Senha incorreta.")
    return False


def main() -> None:
    st.set_page_config(page_title="Assistente de Pós-venda", page_icon="🔧")
    st.title("🔧 Assistente de Pós-venda")

    if not _liberado():
        st.stop()
    ativos = provedores_ativos()
    st.caption(
        "Responde com base no Guia de Onboarding de Pós-venda — cascata "
        f"{' → '.join(ativos) if ativos else '(nenhum provedor)'} "
        "+ embeddings locais + FAISS"
    )

    if not ativos:
        st.error(
            "Nenhuma chave reconhecida. Preencha ao menos uma no `.env`: "
            "`GOOGLE_API_KEY` (Gemini), `GROQ_API_KEY` (Groq), "
            "`OPENAI_API_KEY` (ChatGPT) ou `XAI_API_KEY` (Grok)."
        )

        # Diagnóstico na tela: quase sempre o problema é o arquivo e não a
        # chave — .env numa pasta errada, salvo como .env.txt, ou um valor
        # que ficou depois de um `#` e virou comentário.
        st.markdown("**Diagnóstico**")
        st.code(config.diagnostico(), language=None)

        linhas = []
        for variavel in (
            "GOOGLE_API_KEY",
            "GROQ_API_KEY",
            "OPENAI_API_KEY",
            "XAI_API_KEY",
        ):
            valor = (os.getenv(variavel) or "").strip()
            if not valor:
                linhas.append(f"{variavel}: (vazia)")
            else:
                linhas.append(f"{variavel}: {valor[:6]}... ({len(valor)} caracteres)")
        st.code("\n".join(linhas), language=None)

        st.caption(
            "Se o arquivo existe e as chaves aparecem acima mas a cascata "
            "segue vazia, o prefixo não foi reconhecido: a chave do Gemini "
            "vai em GOOGLE_API_KEY, a da Groq começa com `gsk_` e a da xAI "
            "com `xai-`."
        )
        st.stop()

    cadeia, cadeias_resposta = carregar_cadeia()

    with st.sidebar:
        st.subheader("Cascata")
        for posicao, (rotulo, _) in enumerate(cadeias_resposta, start=1):
            st.caption(f"{posicao}. {rotulo}")
        st.divider()

    if "mensagens" not in st.session_state:
        st.session_state.mensagens = []

    with st.sidebar:
        st.subheader("Perguntas de exemplo")
        for sugestao in SUGESTOES:
            if st.button(sugestao, use_container_width=True):
                st.session_state.pergunta_pendente = sugestao
        st.divider()
        if st.button("Limpar conversa", use_container_width=True):
            st.session_state.mensagens = []
            st.rerun()

    for mensagem in st.session_state.mensagens:
        with st.chat_message(mensagem["papel"]):
            st.markdown(mensagem["texto"])
            if mensagem.get("fontes"):
                with st.expander("Trechos usados do guia"):
                    for fonte in mensagem["fontes"]:
                        st.markdown(f"**{fonte['secao']}**")
                        st.caption(fonte["trecho"])

    pergunta = st.chat_input("Pergunte algo sobre o Pós-venda...")
    if not pergunta:
        pergunta = st.session_state.pop("pergunta_pendente", None)

    if not pergunta:
        return

    st.session_state.mensagens.append({"papel": "user", "texto": pergunta})
    with st.chat_message("user"):
        st.markdown(pergunta)

    # Histórico no formato esperado pelo placeholder do prompt, sem a pergunta atual.
    historico = [
        (("human" if m["papel"] == "user" else "ai"), m["texto"])
        for m in st.session_state.mensagens[:-1]
    ][-8:]

    with st.chat_message("assistant"):
        entrada = cadeia.invoke({"pergunta": pergunta, "historico": historico})

        caixa = st.empty()
        inicio = time.monotonic()
        try:
            texto, trilha = responder_com_cascata(cadeias_resposta, entrada, caixa)
        except Exception as erro:
            caixa.empty()
            st.error(
                "Nenhum provedor da cascata respondeu. Último erro: "
                f"{_motivo(erro)}."
            )
            with st.expander("Detalhes"):
                st.exception(erro)
            st.session_state.mensagens.pop()  # não guarda a pergunta sem resposta
            st.stop()

        segundos = time.monotonic() - inicio
        if len(trilha) > 1:
            # Houve failover: vale mostrar quem caiu e por quê.
            st.caption(" · ".join(trilha) + f" — {segundos:.1f}s")
        else:
            st.caption(f"{trilha[0]} — {segundos:.1f}s")

        fontes = [
            {
                "secao": " > ".join(
                    v
                    for k in ("secao", "subsecao")
                    if (v := doc.metadata.get(k))
                )
                or "Guia de Onboarding",
                "trecho": doc.page_content[:500],
            }
            for doc in entrada["documentos"]
        ]
        with st.expander("Trechos usados do guia"):
            for fonte in fontes:
                st.markdown(f"**{fonte['secao']}**")
                st.caption(fonte["trecho"])

    st.session_state.mensagens.append(
        {"papel": "assistant", "texto": texto, "fontes": fontes}
    )


if __name__ == "__main__":
    main()