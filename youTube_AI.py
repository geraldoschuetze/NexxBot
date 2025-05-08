import streamlit as st
import os
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pytube import YouTube

# Ler variáveis do ambiente
openai_api_key = os.getenv("OPENAI_API_KEY")

# Verificação da chave
if not openai_api_key:
    st.error("🚨 OPENAI_API_KEY não está configurada!")
    st.stop()

# Configuração inicial
st.set_page_config(page_title="YouTube AI Q&A", layout="wide")
st.title("🎥 YouTube AI - Pergunte sobre o vídeo")

# Entradas do usuário
video_url = st.text_input("🔗 Cole a URL do vídeo do YouTube:")
user_question = st.text_input("❓ Faça uma pergunta sobre o vídeo:")

def get_video_transcript(video_url):
    try:
        yt = YouTube(video_url)
        caption = yt.captions.get_by_language_code("pt") or yt.captions.get_by_language_code("en")
        if not caption:
            return None
        return caption.generate_srt_captions()
    except Exception as e:
        st.error(f"Erro ao baixar a legenda: {e}")
        return None

if st.button("🔍 Analisar vídeo"):
    if not video_url or not user_question:
        st.warning("Preencha a URL do vídeo e a pergunta.")
    else:
        with st.spinner("🎬 Extraindo legendas..."):
            transcript = get_video_transcript(video_url)
            if not transcript:
                st.error("⚠️ Não foi possível obter as legendas desse vídeo.")
                st.stop()

        with st.spinner("🤖 Processando com IA..."):
            llm = ChatOpenAI(
                temperature=0.2,
                model_name="gpt-4",
                openai_api_key=openai_api_key,
            )

            # Dividir legenda em trechos
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
            docs = text_splitter.create_documents([transcript])

            # Carregar cadeia de QA
            qa_chain = load_qa_chain(llm, chain_type="stuff")
            response = qa_chain.run(input_documents=docs, question=user_question)

        st.success("✅ Resposta gerada!")
        st.markdown(f"**Pergunta:** {user_question}")
        st.markdown(f"**Resposta:** {response}")
