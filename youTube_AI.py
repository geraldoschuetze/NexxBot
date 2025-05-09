import os
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from pytube import YouTube
import openai
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

# --- Configurações iniciais ---
st.set_page_config(page_title="QA YouTube Dinâmico", layout="wide")
st.title("Análise de Vídeos do YouTube com LangChain + OpenAI")

# 1) Chave OpenAI nos Secrets e configuração da API do OpenAI
api_key = st.secrets.get("OPENAI_API_KEY")
if not api_key:
    st.error("❌ A chave OPENAI_API_KEY não foi encontrada em `st.secrets`.")
    st.stop()
openai.api_key = api_key

# 2) Inputs de URL e Pergunta Dinâmica
url = st.text_input(
    "URL do Vídeo YouTube:",
    placeholder="Cole a URL do vídeo aqui",
    help="Por exemplo: https://www.youtube.com/watch?v=YMJiJWpE-68"
)
question = st.text_input(
    "Pergunta:",
    placeholder="Digite sua pergunta sobre o conteúdo do vídeo",
    help="Qualquer pergunta baseada na transcrição ou áudio do vídeo"
)

# 3) Botão de execução
if st.button("🔍 Analisar"):  
    if not url or not question:
        st.warning("Por favor, insira tanto a URL do vídeo quanto a pergunta.")
    else:
        with st.spinner("Processando... Obtendo legendas/transcrição e consultando OpenAI..."):
            try:
                # Extrai o ID do vídeo da URL
                video_id = url.split("v=")[-1].split("&")[0]
                docs = []

                # 4) Tentativa de obter legendas via YouTubeTranscriptApi
                try:
                    transcripts = YouTubeTranscriptApi.get_transcript(
                        video_id,
                        languages=["pt", "en"],
                        # proxies={'http': '...', 'https': '...'}  # opcional: configure proxies se bloqueado
                    )
                    for snippet in transcripts:
                        docs.append(
                            Document(
                                page_content=snippet['text'],
                                metadata={
                                    'start': snippet['start'],
                                    'duration': snippet['duration']
                                }
                            )
                        )
                except (TranscriptsDisabled, NoTranscriptFound) as yt_err:
                    st.warning("Legendas não disponíveis ou bloqueadas. Usando Whisper para transcrição de áudio...")
                    # 5) Fallback: baixar áudio e usar Whisper
                    yt = YouTube(url)
                    audio_stream = yt.streams.filter(only_audio=True).first()
                    audio_file = audio_stream.download(filename_prefix="yt_audio_")

                    # Chama Whisper via OpenAI
                    with open(audio_file, "rb") as af:
                        transcript = openai.Audio.transcribe("whisper-1", af)
                    text = transcript.get("text", "")
                    docs = [Document(page_content=text, metadata={})]

                # 6) Inicializa modelo e cadeia de QA
                chat = ChatOpenAI(
                    model_name="gpt-3.5-turbo",
                    openai_api_key=api_key
                )
                chain = load_qa_chain(
                    llm=chat,
                    chain_type="map_reduce",
                    verbose=False
                )

                # 7) Executa a pergunta
                result = chain.run(input_documents=docs, question=question)

                # 8) Exibe resultado
                st.subheader("Resposta")
                st.write(result)

            except Exception as e:
                st.error(f"❌ Ocorreu um erro inesperado: {e}")
