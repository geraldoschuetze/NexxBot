import streamlit as st
import os
import openai
import uuid
import mimetypes
from pytube import YouTube
from pydub import AudioSegment
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Chave da OpenAI s
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    st.error("OPENAI_API_KEY não configurada.")
    st.stop()

# Configura layout do app
st.set_page_config(page_title="YouTube AI por Áudio", layout="wide")
st.title("🎧 YouTube AI – Transcreve e responde perguntas")

# Entrada do usuário
video_url = st.text_input("🔗 URL do vídeo do YouTube:")
user_question = st.text_input("❓ Pergunta para a IA:")

MAX_MB = 24  # Limite do Whisper é 25MB

# Função para baixar e converter o áudio
def baixar_audio_do_video(video_url):
    yt = YouTube(video_url)
    audio_stream = yt.streams.filter(only_audio=True).first()
    file_id = str(uuid.uuid4())
    temp_path = f"temp_{file_id}.mp4"
    audio_path = f"temp_{file_id}.mp3"
    audio_stream.download(filename=temp_path)

    # Conversão com qualidade e formato adequado
    audio = AudioSegment.from_file(temp_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(audio_path, format="mp3", bitrate="192k")
    os.remove(temp_path)
    return audio_path

# Função para transcrever e validar o arquivo
def transcrever_audio(audio_path):
    # Verificar tamanho
    tamanho = os.path.getsize(audio_path)
    if tamanho > MAX_MB * 1024 * 1024:
        raise RuntimeError("O arquivo ultrapassa 25MB permitido pelo Whisper da OpenAI.")

    # Verificar tipo MIME
    tipo, _ = mimetypes.guess_type(audio_path)
    if tipo not in ["audio/mpeg", "audio/mp3", "audio/wav", "audio/webm", "audio/mp4", "audio/x-m4a"]:
        raise RuntimeError(f"Tipo de áudio inválido para o Whisper: {tipo}")

    # Enviar para o Whisper
    with open(audio_path, "rb") as f:
        try:
            resultado = openai.Audio.transcribe("whisper-1", f)
            return resultado["text"]
        except Exception as e:
            raise RuntimeError(f"Erro ao transcrever com Whisper: {e}")

# Execução ao clicar no botão
if st.button("🔍 Analisar vídeo"):
    if not video_url or not user_question:
        st.warning("Preencha a URL do vídeo e a pergunta.")
        st.stop()

    with st.spinner("🎧 Baixando e transcrevendo áudio..."):
        try:
            audio_path = baixar_audio_do_video(video_url)
            transcricao = transcrever_audio(audio_path)
            os.remove(audio_path)
        except Exception as e:
            st.error(f"Erro ao processar o áudio: {e}")
            st.stop()

    with st.spinner("🤖 Enviando para ChatGPT..."):
        llm = ChatOpenAI(
            temperature=0.3,
            model_name="gpt-3.5-turbo",
            openai_api_key=openai.api_key,
        )
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.create_documents([transcricao])

        qa_chain = load_qa_chain(llm, chain_type="stuff")
        resposta = qa_chain.run(input_documents=docs, question=user_question)

    st.success("✅ Resposta gerada com sucesso!")
    st.markdown(f"**Pergunta:** {user_question}")
    st.markdown(f"**Resposta:** {resposta}")
