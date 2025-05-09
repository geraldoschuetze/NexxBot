import os
import streamlit as st
from langchain_community.document_loaders import YoutubeLoader
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

# --- Configurações iniciais ---
st.set_page_config(page_title="QA YouTube Dinâmico", layout="wide")
st.title("Análise de Vídeos do YouTube com LangChain + OpenAI")

# 1) Carrega chave da OpenAI dos Secrets
api_key = st.secrets.get("OPENAI_API_KEY")
if not api_key:
    st.error("❌ A chave OPENAI_API_KEY não foi encontrada em `st.secrets`.")
    st.stop()

# Função de leitura via LangChain YoutubeLoader
def carrega_youtube(video_id):
    loader = YoutubeLoader(
        video_id,
        add_video_info=False,
        language=["pt", "en"]
    )
    lista_documentos = loader.load()
    # junta fragmentos em um único texto
    documento = "\n\n".join([doc.page_content for doc in lista_documentos])
    return documento

# 2) Inputs de URL e Pergunta Dinâmica
url = st.text_input(
    "URL do Vídeo YouTube:",
    placeholder="Cole a URL do vídeo aqui",
    help="Por exemplo: https://www.youtube.com/watch?v=YMJiJWpE-68"
)
question = st.text_input(
    "Pergunta:",
    placeholder="Digite sua pergunta sobre o conteúdo do vídeo",
    help="Qualquer pergunta baseada na transcrição das legendas do vídeo"
)

# 3) Botão de execução
if st.button("🔍 Analisar"):  
    if not url or not question:
        st.warning("Por favor, insira tanto a URL do vídeo quanto a pergunta.")
    else:
        with st.spinner("Processando... Carregando transcript e consultando OpenAI..."):
            try:
                # extrai ID do vídeo
                video_id = url.split("v=")[-1].split("&")[0]
                # carrega o texto completo do vídeo
                texto = carrega_youtube(video_id)
                # encapsula em Document para LangChain
                docs = [Document(page_content=texto, metadata={"source": video_id})]

                # inicializa modelo e cadeia de QA
                chat = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=api_key)
                chain = load_qa_chain(llm=chat, chain_type="map_reduce", verbose=False)

                # executa a pergunta
                result = chain.run(input_documents=docs, question=question)

                # exibe resultado
                st.subheader("Resposta")
                st.write(result)
            except Exception as e:
                st.error(f"❌ Ocorreu um erro inesperado: {e}")
