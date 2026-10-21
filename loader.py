"""
Leitura do documento .docx e conversão para Markdown.

Por que não usar o Docx2txtLoader do LangChain: ele joga fora a estrutura das
tabelas, e neste guia as tabelas são justamente a parte mais consultada
(pesos do score, régua de faróis, onde encontrar cada indicador). Aqui
percorremos o corpo do documento na ordem real e convertemos cada tabela em
Markdown, para que o modelo consiga ler linha a linha.
"""

from __future__ import annotations

from pathlib import Path

from docx import Document as DocxDocument
from docx.table import Table
from docx.text.paragraph import Paragraph


def _iter_blocos(doc: DocxDocument):
    """Percorre parágrafos e tabelas na ordem em que aparecem no documento."""
    corpo = doc.element.body
    for filho in corpo.iterchildren():
        tag = filho.tag.split("}")[-1]
        if tag == "p":
            yield Paragraph(filho, doc)
        elif tag == "tbl":
            yield Table(filho, doc)


def _nivel_titulo(paragrafo: Paragraph) -> int:
    """Retorna 1 ou 2 se o parágrafo for um Heading, senão 0."""
    estilo = paragrafo.style.name if paragrafo.style is not None else ""
    for nivel in (1, 2, 3):
        if estilo in (f"Heading {nivel}", f"Título {nivel}"):
            return nivel
    return 0


def _celula(texto: str) -> str:
    """Normaliza o texto de uma célula para caber numa linha de Markdown."""
    return " ".join(texto.split()).replace("|", "/")


def _tabela_para_markdown(tabela: Table) -> str:
    linhas = [[_celula(c.text) for c in linha.cells] for linha in tabela.rows]
    if not linhas:
        return ""

    cabecalho, *resto = linhas
    largura = max(len(l) for l in linhas)
    cabecalho += [""] * (largura - len(cabecalho))

    saida = ["| " + " | ".join(cabecalho) + " |"]
    saida.append("| " + " | ".join(["---"] * largura) + " |")
    for linha in resto:
        linha = linha + [""] * (largura - len(linha))
        saida.append("| " + " | ".join(linha) + " |")
    return "\n".join(saida)


def docx_para_markdown(caminho: str | Path) -> str:
    """Converte um .docx em Markdown preservando títulos, listas e tabelas."""
    doc = DocxDocument(str(caminho))
    partes: list[str] = []

    for bloco in _iter_blocos(doc):
        if isinstance(bloco, Table):
            md = _tabela_para_markdown(bloco)
            if md:
                partes.append(md)
            continue

        texto = bloco.text.strip()
        if not texto:
            continue

        nivel = _nivel_titulo(bloco)
        estilo = bloco.style.name if bloco.style is not None else ""

        if nivel:
            partes.append("#" * nivel + " " + texto)
        elif estilo == "List Paragraph":
            partes.append("- " + texto)
        else:
            partes.append(texto)

    return "\n\n".join(partes)


if __name__ == "__main__":
    import sys

    caminho = sys.argv[1] if len(sys.argv) > 1 else "data/Guia_Onboarding_Pos-venda.docx"
    print(docx_para_markdown(caminho))
