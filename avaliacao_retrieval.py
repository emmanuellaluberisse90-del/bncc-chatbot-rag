
import streamlit as st
import pandas as pd

from rag_engine import (
    conectar_servicos,
    buscar_resultados_avaliacao
)

st.set_page_config(
    page_title="Avaliação do Retrieval",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 Avaliação do Retrieval — BNCC Computação")

st.write(
    "Teste inicial da recuperação de documentos "
    "utilizando diretamente o Pinecone, sem o filtro de relevância do Gemini."
)

# Pergunta de teste
pergunta = (
    "Segundo a BNCC Computação, o que significa "
    "organizar objetos considerando padrões e diferenças?"
)

documento = (
    "Anexo ao Parecer CNECEB nº 2-2022 - BNCC - Computação (1).pdf"
)

st.subheader("Pergunta de teste")
st.info(pergunta)

st.caption(f"Documento: {documento}")

if st.button("Executar busca", type="primary"):

    with st.spinner("Consultando o Pinecone..."):

        try:
            # Conectar aos serviços
            client, index = conectar_servicos()

            # Recuperar os 5 melhores resultados
            resultados = buscar_resultados_avaliacao(
                client=client,
                index=index,
                pergunta=pergunta,
                top_k=5,
                documento_forcado=documento
            )

            if resultados:

                st.success(
                    f"{len(resultados)} resultados recuperados."
                )

                dados = []

                for posicao, resultado in enumerate(resultados, start=1):

                    dados.append({
                        "Posição": posicao,
                        "ID": resultado["id"],
                        "Documento": resultado["documento"],
                        "Similaridade": round(
                            resultado["similaridade"], 4
                        ),
                        "Texto": resultado["texto"]
                    })

                df = pd.DataFrame(dados)

                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True
                )

            else:
                st.warning("Nenhum resultado foi encontrado.")

        except Exception as erro:
            st.error(f"Erro durante a avaliação: {erro}")
