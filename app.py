
import streamlit as st

from rag_engine import (
    conectar_servicos,
    identificar_documento,
    buscar_contexto_relevante,
    MODELO_GERACAO
)


# ==========================================
# CONFIGURAÇÃO DA PÁGINA
# ==========================================

st.set_page_config(
    page_title="BNCC Computação - Chatbot",
    page_icon="📚",
    layout="centered"
)

st.title("📚 BNCC Computação")
st.caption(
    "Chatbot para consulta de documentos de Computação na Educação."
)


# ==========================================
# CONEXÃO COM GEMINI E PINECONE
# ==========================================

@st.cache_resource
def carregar_servicos():
    return conectar_servicos()


try:
    client, index = carregar_servicos()

except Exception as erro:
    st.error(f"Erro ao conectar aos serviços: {erro}")
    st.stop()


# ==========================================
# MEMÓRIA DA CONVERSA
# ==========================================

if "mensagens" not in st.session_state:
    st.session_state.mensagens = []

if "ultimo_documento" not in st.session_state:
    st.session_state.ultimo_documento = None


# ==========================================
# NOVA CONVERSA
# ==========================================

if st.button("🆕 Nova conversa"):

    st.session_state.mensagens = []
    st.session_state.ultimo_documento = None

    st.rerun()


# ==========================================
# EXIBIR HISTÓRICO
# ==========================================

for mensagem in st.session_state.mensagens:

    with st.chat_message(mensagem["role"]):
        st.markdown(mensagem["content"])


# ==========================================
# TRANSFORMAR PERGUNTA EM CONTEXTO COMPLETO
# ==========================================

def contextualizar_pergunta(pergunta):

    historico = st.session_state.mensagens[-8:]

    if not historico:
        return pergunta

    conversa = "\n".join([
        f"{item['role']}: {item['content']}"
        for item in historico
    ])

    prompt = f"""
Você é um assistente que interpreta perguntas
de acompanhamento em uma conversa sobre
Computação na Educação.

HISTÓRICO DA CONVERSA:
{conversa}

NOVA PERGUNTA:
{pergunta}

INSTRUÇÕES:
- Entenda a nova pergunta usando o histórico.
- Se houver expressões como "essas habilidades",
  "esse documento" ou "e os objetivos?", recupere
  o assunto correspondente no histórico.
- Transforme a pergunta em uma pergunta completa
  e independente.
- Preserve o ano escolar, assunto e documento
  mencionados anteriormente quando forem relevantes.
- Não invente informações.
- Não responda à pergunta.
- Não acrescente um documento que não foi mencionado.

Retorne somente a pergunta contextualizada.
"""

    resposta = client.models.generate_content(
        model=MODELO_GERACAO,
        contents=prompt
    )

    return resposta.text.strip()


# ==========================================
# GERAR RESPOSTA COM BASE NOS DOCUMENTOS
# ==========================================

def gerar_resposta(pergunta, contexto):

    historico = st.session_state.mensagens[-8:]

    conversa = "\n".join([
        f"{item['role']}: {item['content']}"
        for item in historico
    ])

    trechos = "\n\n".join([
        f"""
DOCUMENTO: {item['documento']}
TRECHO:
{item['texto']}
"""
        for item in contexto
    ])

    prompt = f"""
Você é um assistente especializado em
Computação na Educação.

Sua tarefa é responder à pergunta do usuário
com base exclusivamente nos trechos recuperados
dos documentos.

HISTÓRICO DA CONVERSA:
{conversa}

PERGUNTA ATUAL:
{pergunta}

TRECHOS RECUPERADOS:
{trechos}

REGRAS OBRIGATÓRIAS:

1. Responda somente com informações presentes
   nos trechos recuperados.

2. Não invente habilidades, códigos, objetivos,
   competências ou informações.

3. Preserve os códigos e a identificação
   dos documentos exatamente como aparecem.

4. Não misture habilidades de documentos
   ou currículos diferentes.

5. Considere o ano escolar solicitado.

6. Se os trechos não contiverem informações
   suficientes para responder, informe claramente
   que a informação não foi localizada nos trechos
   disponíveis.

7. Quando possível, identifique o documento
   utilizado na resposta.

8. Use português brasileiro, com linguagem clara
   e organizada.

Responda diretamente à pergunta.
"""

    resposta = client.models.generate_content(
        model=MODELO_GERACAO,
        contents=prompt
    )

    return resposta.text.strip()


# ==========================================
# RECEBER NOVA PERGUNTA
# ==========================================

pergunta = st.chat_input(
    "Digite sua pergunta sobre Computação na Educação..."
)


if pergunta:

    # Exibir pergunta do usuário
    st.session_state.mensagens.append({
        "role": "user",
        "content": pergunta
    })

    with st.chat_message("user"):
        st.markdown(pergunta)

    with st.chat_message("assistant"):

        try:

            with st.spinner("Analisando sua pergunta..."):

                # 1. Verificar se o usuário mencionou
                # explicitamente um documento
                documento_atual = identificar_documento(pergunta)

                if documento_atual:

                    st.session_state.ultimo_documento = documento_atual

                # 2. Contextualizar pergunta de acompanhamento
                pergunta_contextualizada = contextualizar_pergunta(
                    pergunta
                )

                # 3. Definir documento da busca
                documento_forcado = (
                    documento_atual
                    or st.session_state.ultimo_documento
                )

                # 4. Buscar trechos relevantes
                contexto = buscar_contexto_relevante(
                    client,
                    index,
                    pergunta_contextualizada,
                    top_k=10 if documento_forcado else 5,
                    documento_forcado=documento_forcado
                )

                # 5. Verificar se há contexto relevante
                if not contexto:

                    resposta_final = (
                        "Não encontrei informações relevantes "
                        "nos documentos disponíveis para responder "
                        "a esta pergunta."
                    )

                else:

                    # 6. Gerar resposta fundamentada
                    resposta_final = gerar_resposta(
                        pergunta_contextualizada,
                        contexto
                    )

            st.markdown(resposta_final)

            # Exibir documentos consultados
            if contexto:

                documentos_utilizados = list(dict.fromkeys([
                    item["documento"] for item in contexto
                ]))

                with st.expander("📄 Documentos consultados"):

                    for documento in documentos_utilizados:
                        st.write(f"- {documento}")

            # Salvar resposta no histórico
            st.session_state.mensagens.append({
                "role": "assistant",
                "content": resposta_final
            })

        except Exception as erro:

            st.error(
                f"Ocorreu um erro ao processar sua pergunta: {erro}"
            )
