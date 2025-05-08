import streamlit as st
import os
import openai
import uuid
import mimetypes
from pytube import YouTube
from pydub import AudioSegment
from pathlib import Path
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Carrega API Key da OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    st.error("OPENAI_API_KEY não configurada.")
    st.stop()

# Layout da página
st.set_page_config(page_title="YouTube AI por Áudio", layout="wide")
st.title("🎧 YouTube AI – Transcreve e responde perguntas")

video_url = st.text_input("🔗 URL do vídeo do YouTube:")
user_question = st.text_input("❓ Pergunta para a IA:")

MAX_MB = 24  # Limite do Whisper

def baixar_audio_do_video(video_url):
    yt = YouTube(video_url)
    audio_stream = yt.streams.filter(only_audio=True).first()
    file_id = str(uuid.uuid4())
    temp_path = f"temp_{file_id}.mp4"
    audio_path = f"temp_{file_id}.wav"
    audio_stream.download(filename=temp_path)

    # Conversão para .wav com 16kHz e mono
    audio = AudioSegment.from_file(temp_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(audio_path, format="wav")
    os.remove(temp_path)
    return audio_path

def transcrever_audio(audio_path):
    tamanho_mb = os.path.getsize(audio_path) / 1024 / 1024
    duracao = AudioSegment.from_file(audio_path).duration_seconds
    tipo_mime, _ = mimetypes.guess_type(audio_path)

    # Exibir debug no app
    st.code(f"""
[DEBUG]
Arquivo: {Path(audio_path).name}
Tamanho: {tamanho_mb:.2f} MB
Duração: {duracao:.1f} segundos
MIME: {tipo_mime}
""")

    if tamanho_mb > MAX_MB:
        raise RuntimeError("O arquivo de áudio ultrapassa o limite de 25MB.")
    if tipo_mime not in ["audio/wav", "audio/x-wav"]:
        raise RuntimeError(f"Tipo de áudio inválido para o Whisper: {tipo_mime}")

    with open(audio_path, "rb") as f:
        try:
            response = openai.Audio.transcribe("whisper-1", f)
            return response["text"]
        except Exception as e:
            raise RuntimeError(f"Erro na transcrição com Whisper: {e}")

if st.button("🔍 Analisar vídeo"):
    if not video_url or not user_question:
        st.warning("Preencha a URL e a pergunta.")
        st.stop()

    with st.spinner("🔊 Baixando e transcrevendo o áudio..."):
        try:
            audio_path = baixar_audio_do_video(video_url)
            transcricao = transcrever_audio(audio_path)
            os.remove(audio_path)
        except Exception as e:
            st.error(f"Erro ao processar o áudio: {e}")
            st.stop()

    with st.spinner("🤖 Enviando para o ChatGPT..."):
        llm = ChatOpenAI(
            temperature=0.3,
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
