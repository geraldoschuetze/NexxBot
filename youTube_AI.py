import streamlit as st
import os
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Ler variáveis do ambiente (definidas no Streamlit Secrets)
openai_api_key = os.getenv("OPENAI_API_KEY")
langsmith_key = os.getenv("LANGSMITH_API_KEY")
langsmith_endpoint = os.getenv("LANGSMITH_ENDPOINT")
langsmith_project = os.getenv("LANGSMITH_PROJECT")
langsmith_tracing = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"

# Configuração inicial do Streamlit
st.set_page_config(page_title="YouTube AI", layout="wide")
st.title("🎥 YouTube Video Analyzer with AI")

# Verificar se as chaves foram carregadas corretamente
if not openai_api_key:
    st.error("🚨 OPENAI_API_KEY não está configurada!")
    st.stop()

# Inicializar LLM com OpenAI
llm = ChatOpenAI(
    temperature=0,
    model_name="gpt-4",
    openai_api_key=openai_api_key,
)

# Prompt de exemplo
prompt = PromptTemplate(
    input_variables=["video_title"],
    template="Resuma o vídeo com título '{video_title}' e explique os principais tópicos de forma simples."
)

chain = LLMChain(llm=llm, prompt=prompt)

# Interface principal
video_title = st.text_input("🔍 Título do vídeo", "")

if st.button("Analisar"):
    if not video_title:
        st.warning("Por favor, insira um título de vídeo.")
    else:
        with st.spinner("Analisando com IA..."):
            result = chain.run(video_title=video_title)
            st.success("✅ Análise concluída!")
            st.write(result)
