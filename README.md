# Assistente de Pós-venda — Agente RAG com cascata de provedores

Chatbot que responde perguntas sobre o **Guia de Onboarding de Pós-venda**
usando RAG: o documento é indexado num banco vetorial e só o que é relevante
para cada pergunta vai para o modelo. Mesma arquitetura da aula de agentes RAG
da Hashtag, com duas diferenças: o documento é um **.docx** e o chat não
depende de um único fornecedor.

```
Streamlit  →  LangChain (LCEL)  →  Gemini ──falhou?──▶ Groq ──falhou?──▶ ChatGPT ──▶ Grok
                    ↑                       (~200 ms)        (~200 ms)
              FAISS + embeddings locais (sentence-transformers)
                    ↑
              Guia de Onboarding (.docx)
```

## Como rodar

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env             # Windows: copy .env.example .env
# edite o .env e preencha pelo menos uma chave. Gratuitas: GOOGLE_API_KEY
# (Gemini) e GROQ_API_KEY (Groq). Pagas: OPENAI_API_KEY e XAI_API_KEY.

python ingest.py                 # gera o índice (uma vez só)
streamlit run app.py
```

Na primeira execução o `ingest.py` baixa o modelo de embedding (~470 MB) para
o cache do Hugging Face. Depois disso ele roda offline.

Teste rápido sem interface:

```bash
python perguntar.py "Quais são as faixas de farol do agendamento?"
python modelos.py                # lista os modelos de cada chave configurada
```

## A cascata

`provedor.py` monta a lista de tentativas a partir das chaves que existem no
`.env`, na ordem de `ORDEM_PROVEDORES`. Cada provedor pode ter mais de um
modelo, e cada par provedor+modelo é um degrau:

```
1. Gemini · gemini-flash-latest     4. Groq · openai/gpt-oss-120b
2. Gemini · gemini-3.8-flash        5. Groq · llama-3.1-8b-instant
3. Groq · llama-3.3-70b-versatile   (+ ChatGPT e Grok, se houver chave)
```

A ordem padrão começa pelos gratuitos. **Gemini** (AI Studio) e **Groq** têm
camada gratuita; **OpenAI** e **xAI** cobram por uso — e a assinatura do
ChatGPT ou do Grok no aplicativo não dá crédito de API, são produtos
separados.

Quando um degrau falha — 429, 503, 404, timeout, chave rejeitada — o próximo
assume depois de `ESPERA_FALHA_MS` (200 ms). Abaixo da resposta aparece quem
respondeu e quem caiu no caminho, com o motivo e o tempo total.

**Sobre "milissegundos":** a troca entre provedores é de milissegundos, mas a
resposta de um LLM leva segundos. O que define quanto se espera por um
provedor antes de desistir dele é `TIMEOUT_SEGUNDOS` (20s por padrão). Baixo
demais corta respostas boas pela metade; alto demais faz você esperar por um
serviço que já morreu.

Não há retry no mesmo degrau: se o provedor falhou, o seguinte tem muito mais
chance de responder rápido do que o mesmo tentando de novo. É o oposto do
retry com backoff — aqui a redundância está na diversidade de fornecedores,
não na insistência.

## Os arquivos

| Arquivo | O que faz |
| --- | --- |
| `loader.py` | Lê o `.docx` e converte para Markdown preservando títulos, listas e **tabelas**. |
| `ingest.py` | Quebra em pedaços, gera os embeddings e salva o índice FAISS em `indice_faiss/`. |
| `embeddings.py` | Modelo de embedding local, compartilhado entre a indexação e a busca. |
| `app.py` | Interface de chat no Streamlit, com histórico, failover e os trechos usados. |
| `provedor.py` | Monta a cascata de provedores a partir das chaves do `.env`. |
| `perguntar.py` | Mesma cadeia pelo terminal. |
| `modelos.py` | Lista os modelos de chat disponíveis em cada provedor configurado. |
| `data/` | O documento-fonte. |

## Decisões de projeto

**Embedding local, chat na nuvem.** A busca roda na sua máquina com
`sentence-transformers`: sem chave, sem cota e sem 503. O modelo é multilíngue
porque o guia está em português — modelos só de inglês degradam bastante a
busca aqui. Para um documento deste tamanho, indexar leva segundos na CPU. O
índice depende só de `MODELO_EMBEDDING`, então trocar de provedor de chat não
obriga a reindexar.

**Erros classificados pelo conteúdo, não pela classe.** Cada integração
reembrulha o erro do SDK numa exceção própria (`GoogleAPIError`,
`APIStatusError`...), então capturar a exceção original não é confiável. O app
procura os sinais no texto: 429/503 é sobrecarga, 404 é modelo descontinuado,
401/403 é chave. Isso mantém o failover funcionando quando as bibliotecas
mudam de versão.

**Tabelas convertidas para Markdown.** O `Docx2txtLoader` do LangChain achata
as tabelas e perde a estrutura. Neste guia, as tabelas são a parte mais
consultada — pesos do score, régua de faróis, onde encontrar cada indicador —
então `loader.py` percorre o corpo do documento na ordem real e escreve cada
tabela em Markdown. Sem isso, "qual o peso do faturamento no score?" vira
chute.

**Quebra em dois passos.** Primeiro por título (`MarkdownHeaderTextSplitter`),
depois por tamanho (`RecursiveCharacterTextSplitter`, 1200/200). Cada pedaço
carrega na metadata a seção de origem e é prefixado com ela, o que melhora a
busca e permite citar a fonte na resposta.

**Cabeçalho de tabela reposto.** Duas tabelas (3.1 e o glossário) não cabem em
um pedaço só. Quando isso acontece, a continuação recebe de volta a linha de
cabeçalho — senão o modelo vê linhas soltas sem saber o que é cada coluna.

**Prompt fechado no contexto.** O system prompt proíbe conhecimento geral,
manda admitir quando o guia não cobre o assunto e exige citar números
exatamente como estão no documento. É o que separa um RAG confiável de um
modelo que inventa um peso plausível.

## Trocar o documento

Coloque o `.docx` em `data/`, aponte `CAMINHO_DOCUMENTO` no `.env` e rode
`python ingest.py --forcar`. Para PDF, troque o `loader.py` por `PyPDFLoader`;
o resto da cadeia não muda.

## Trocar ou acrescentar provedor

Tudo que é específico de fornecedor mora em `provedor.py`: as chaves, os
endpoints e as fábricas de cliente. Para acrescentar um quarto provedor
compatível com a API da OpenAI, basta uma entrada em `MODELOS`, uma em
`ROTULOS` e o `base_url` certo — o resto da cadeia não muda.
