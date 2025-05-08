import streamlit as st
import os
import openai
from pytube import YouTube
from pydub import AudioSegment
import uuid
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Carregar chave da OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    st.error("OPENAI_API_KEY não está configurada.")
    st.stop()

st.set_page_config(page_title="YouTube AI com Transcrição", layout="wide")
st.title("🎧 YouTube AI – Pergunte sobre o vídeo (com transcrição automática)")

video_url = st.text_input("🔗 Cole a URL do vídeo do YouTube:")
user_question = st.text_input("❓ Faça uma pergunta sobre o vídeo:")

def transcrever_audio(file_path):
    with open(file_path, "rb") as f:
        transcript = openai.Audio.transcribe("whisper-1", f)
    return transcript["text"]

def baixar_audio_do_video(video_url):
    yt = YouTube(video_url)
    audio_stream = yt.streams.filter(only_audio=True).first()
    file_id = str(uuid.uuid4())
    temp_path = f"temp_{file_id}.mp4"
    audio_path = f"temp_{file_id}.mp3"
    audio_stream.download(filename=temp_path)

    # Converter para mp3
    audio = AudioSegment.from_file(temp_path)
    audio.export(audio_path, format="mp3")
    os.remove(temp_path)
    return audio_path

if st.button("🔍 Analisar vídeo"):
    if not video_url or not user_question:
        st.warning("Preencha a URL do vídeo e a pergunta.")
        st.stop()

    with st.spinner("🔊 Baixando e transcrevendo áudio..."):
        try:
            audio_path = baixar_audio_do_video(video_url)
            transcricao = transcrever_audio(audio_path)
            os.remove(audio_path)
        except Exception as e:
            st.error(f"Erro durante transcrição: {e}")
            st.stop()

    with st.spinner("🤖 Analisando com ChatGPT..."):
        llm = ChatOpenAI(
            temperature=0.2,
            model_name="gpt-3.5-turbo",
            openai_api_key=openai.api_key,
        )
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.create_documents([transcricao])

        qa_chain = load_qa_chain(llm, chain_type="stuff")
        resposta = qa_chain.run(input_documents=docs, question=user_question)

    st.success("✅ Resposta gerada!")
    st.markdown(f"**Pergunta:** {user_question}")
    st.markdown(f"**Resposta:** {resposta}")
