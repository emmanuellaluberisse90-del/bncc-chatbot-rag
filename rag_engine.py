import streamlit as st
import os
import json
import unicodedata

from google import genai
from google.genai import types
from pinecone import Pinecone


# ==========================================
# CONFIGURAÇÕES
# ==========================================

MODELO_EMBEDDING = "gemini-embedding-001"
MODELO_GERACAO = "gemini-3.1-flash-lite"

NOME_INDEX = "bncc-computacao-v2"
NAMESPACE = "bncc-computacao"

# ==========================================
# CONEXÕES
# ==========================================


def conectar_servicos():

    gemini_key = st.secrets.get(
        "GEMINI_API_KEY",
        os.getenv("GEMINI_API_KEY")
    )

    pinecone_key = st.secrets.get(
        "PINECONE_API_KEY",
        os.getenv("PINECONE_API_KEY")
    )

    if not gemini_key or not pinecone_key:
        raise ValueError(
            "Configure GEMINI_API_KEY e PINECONE_API_KEY "
            "nas configurações de Secrets."
        )

    client = genai.Client(api_key=gemini_key)

    pc = Pinecone(api_key=pinecone_key)
    index = pc.Index(NOME_INDEX)

    return client, index


# ==========================================
# NORMALIZAÇÃO DE TEXTO
# ==========================================

def normalizar_texto(texto):

    texto = texto.lower().strip()
    texto = unicodedata.normalize("NFD", texto)

    texto = "".join(
        c for c in texto
        if unicodedata.category(c) != "Mn"
    )

    return " ".join(texto.split())


# ==========================================
# IDENTIFICAÇÃO DOS DOCUMENTOS
# ==========================================

documentos_alias = {

    "Anexo ao Parecer CNECEB nº 2-2022 - BNCC - Computação (1).pdf": [
        "bncc computacao",
        "bncc de computacao",
        "anexo ao parecer cne",
        "parecer cne 2 2022",
        "documento bncc"
    ],

    "Resolução CEE-PA n 577-2025 - Apensos_Parecer CEE-PA n 603-2025 e Referencial Curricular da Computação-SEDUC (1) (1).pdf": [
        "resolucao do para",
        "resolucao cee pa",
        "cee pa 577 2025",
        "referencial curricular do para",
        "referencial da computacao do para"
    ],

    "deliberacao_04_25_ok (1) (1).pdf": [
        "deliberacao",
        "deliberacao 04 25",
        "documento da deliberacao"
    ],

    "ilovepdf_merged (1) (1) (1).pdf": [
        "documento compilado",
        "arquivo compilado",
        "documento ilovepdf"
    ]
}


def identificar_documento(pergunta):

    pergunta_normalizada = normalizar_texto(pergunta)

    for documento, apelidos in documentos_alias.items():

        for apelido in apelidos:

            if normalizar_texto(apelido) in pergunta_normalizada:
                return documento

    return None


# ==========================================
# BUSCA INTELIGENTE NO PINECONE
# ==========================================

def buscar_documentos_inteligente(
    client,
    index,
    pergunta,
    top_k=5,
    documento_forcado=None
):

    documento = documento_forcado or identificar_documento(pergunta)

    if documento:

        documentos_busca = [documento]

    else:

        documentos_busca = list(documentos_alias.keys())

    resultado = client.models.embed_content(
        model=MODELO_EMBEDDING,
        contents=pergunta,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY"
        )
    )

    embedding_pergunta = resultado.embeddings[0].values

    resultados_por_documento = {}

    for doc in documentos_busca:

        resposta = index.query(
            namespace=NAMESPACE,
            vector=embedding_pergunta,
            top_k=top_k,
            include_metadata=True,
            filter={"documento": {"$eq": doc}}
        )

        resultados_por_documento[doc] = resposta.matches

    return resultados_por_documento


# ==========================================
# AVALIAÇÃO DE RELEVÂNCIA EM LOTE
# ==========================================

def avaliar_relevancia_lote(client, pergunta, resultados):

    trechos = []

    for documento, matches in resultados.items():

        for item in matches:

            trechos.append({
                "id": item.id,
                "documento": documento,
                "texto": item.metadata.get("texto", "")
            })

    if not trechos:
        return {}

    conteudo = "\n\n".join([
        f"""
ID: {item['id']}
DOCUMENTO: {item['documento']}
TRECHO: {item['texto']}
"""
        for item in trechos
    ])

    prompt = f"""
Você é um avaliador de relevância para um sistema RAG
sobre Computação na Educação.

PERGUNTA DO USUÁRIO:
{pergunta}

Analise cada trecho separadamente.

REGRAS:
- Considere relevante somente o trecho que contém informações
  úteis para responder diretamente à pergunta.
- Não considere apenas a semelhança de palavras.
- Não invente informações.
- Considere o ano escolar e o assunto solicitado.
- Se o trecho tratar de outro ano ou assunto sem relação,
  considere irrelevante.
- Preserve a identificação de cada documento.
- Avalie todos os trechos apresentados.

TRECHOS:
{conteudo}

Responda exclusivamente em JSON, neste formato:
{{
  "resultados": [
    {{"id": "ID_DO_TRECHO", "relevante": true}},
    {{"id": "ID_DO_TRECHO", "relevante": false}}
  ]
}}
"""

    resposta = client.models.generate_content(
        model=MODELO_GERACAO,
        contents=prompt
    )

    texto_resposta = resposta.text.strip()
    texto_resposta = texto_resposta.replace("```json", "")
    texto_resposta = texto_resposta.replace("```", "").strip()

    dados = json.loads(texto_resposta)

    return {
        item["id"]: item["relevante"]
        for item in dados["resultados"]
    }


# ==========================================
# MONTAGEM DO CONTEXTO RELEVANTE
# ==========================================

def buscar_contexto_relevante(
    client,
    index,
    pergunta,
    top_k=5,
    documento_forcado=None
):

    resultados = buscar_documentos_inteligente(
        client,
        index,
        pergunta,
        top_k=top_k,
        documento_forcado=documento_forcado
    )

    relevancia = avaliar_relevancia_lote(
        client,
        pergunta,
        resultados
    )

    contexto_relevante = []

    for documento, trechos in resultados.items():

        for trecho in trechos:

            if relevancia.get(trecho.id, False):

                contexto_relevante.append({
                    "documento": documento,
                    "id": trecho.id,
                    "similaridade": trecho.score,
                    "texto": trecho.metadata.get("texto", "")
                })

    return contexto_relevante
