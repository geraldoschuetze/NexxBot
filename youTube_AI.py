import os
import sys
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

# Carrega variáveis de ambiente
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

# Lê a chave da API da OpenAI
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("A variável de ambiente OPENAI_API_KEY não está definida.")
    st.stop()

# Título do app
st.title("🔍 Analisador de Vídeo do YouTube com IA")

# Inputs do usuário
url = st.text_input("URL do vídeo no YouTube:")
question = st.text_area("Digite sua pergunta para a IA:")

# Botão para processar
if st.button("Analisar Vídeo"):
    if not url or not question:
        st.warning("Por favor, preencha a URL e a pergunta.")
        st.stop()

    with st.spinner("Carregando transcrição do vídeo..."):
        try:
            loader = YoutubeLoader.from_youtube_url(
                url,
                add_video_info=False,
                language=["pt", "en"],
                translation=None
            )
            docs = loader.load()
        except Exception as e:
            st.error(f"Erro ao carregar vídeo: {e}")
            st.stop()

    with st.spinner("Processando pergunta com IA..."):
        try:
            chat = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                openai_api_key=api_key
            )

            chain = load_qa_chain(
                llm=chat,
                chain_type="map_reduce",  # melhor para longos
                verbose=False
            )

            answer = chain.run(input_documents=docs, question=question)
            st.success("Resposta gerada com sucesso!")
            st.markdown("### 🧠 Resposta da IA:")
            st.write(answer)
        except Exception as e:
            st.error(f"Erro ao processar pergunta: {e}")
