import streamlit as st
import os
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

# API Key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OPENAI_API_KEY não configurada.")
    st.stop()

# Layout da página
st.set_page_config(page_title="YouTube AI com Transcrição do Próprio Vídeo", layout="wide")
st.title("🎥 YouTube AI – Pergunte sobre o vídeo")

# Inputs do usuário
video_url = st.text_input("🔗 URL do vídeo do YouTube:")
user_question = st.text_input("❓ Pergunta para a IA sobre o vídeo:")

if st.button("🔍 Analisar vídeo"):
    if not video_url or not user_question:
        st.warning("Preencha a URL e a pergunta.")
        st.stop()

    with st.spinner("📄 Extraindo transcrição do vídeo..."):
        try:
            loader = YoutubeLoader.from_youtube_url(
                video_url,
                add_video_info=False,
                language=["pt", "en"],
                translation=None
            )
            docs = loader.load()
        except Exception as e:
            st.error(f"Erro ao carregar legendas: {e}")
            st.stop()

    with st.spinner("🤖 Enviando para o modelo GPT..."):
        chat = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            openai_api_key=openai_api_key,
        )
        chain = load_qa_chain(llm=chat, chain_type="map_reduce", verbose=False)
        resposta = chain.run(input_documents=docs, question=user_question)

    st.success("✅ Resposta gerada!")
    st.markdown(f"**Pergunta:** {user_question}")
    st.markdown(f"**Resposta:** {resposta}")
