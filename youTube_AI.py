import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

# --- Configurações iniciais ---
st.set_page_config(page_title="QA YouTube Dinâmico", layout="wide")
st.title("Análise de Vídeos do YouTube com LangChain + OpenAI")

# 1) Carrega a chave da OpenAI do Streamlit Secrets
api_key = st.secrets.get("OPENAI_API_KEY")
if not api_key:
    st.error("❌ A chave OPENAI_API_KEY não foi encontrada em `st.secrets`.")
    st.stop()

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
        with st.spinner("Processando... Obtendo legendas e consultando a OpenAI..."):
            try:
                # Extrai o ID do vídeo da URL
                video_id = url.split("v=")[-1].split("&")[0]

                # Busca as legendas em PT e EN
                transcripts = YouTubeTranscriptApi.get_transcript(
                    video_id,
                    languages=["pt", "en"]
                )

                # Converte para Document do LangChain
                docs = [
                    Document(
                        page_content=snippet['text'],
                        metadata={'start': snippet['start'], 'duration': snippet['duration']}
                    )
                    for snippet in transcripts
                ]

                # Inicializa o modelo e a cadeia de QA
                chat = ChatOpenAI(
                    model_name="gpt-3.5-turbo",
                    openai_api_key=api_key
                )
                chain = load_qa_chain(
                    llm=chat,
                    chain_type="map_reduce",
                    verbose=False
                )

                # Executa a pergunta sobre qualquer tópico
                result = chain.run(input_documents=docs, question=question)

                # Exibe o resultado
                st.subheader("Resposta")
                st.write(result)

            except Exception as e:
                st.error(f"❌ Ocorreu um erro: {e}")
