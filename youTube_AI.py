import streamlit as st
from urllib.parse import urlparse, parse_qs
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document

# Lê a chave da OpenAI dos secrets do Streamlit
# O secret está definido como OPENAI_API_KEY
gpt_api_key = st.secrets.get("OPENAI_API_KEY")
if not gpt_api_key:
    st.error("A chave OPENAI_API_KEY não está configurada nos secrets.")
    st.stop()

# Inicializa modelo e cadeia de QA
chat = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=gpt_api_key)
chain = load_qa_chain(llm=chat, chain_type="map_reduce", verbose=True)

# Função para carregar e concatenar legendas do YouTube
def carrega_youtube(video_id):
    loader = YoutubeLoader(video_id, add_video_info=False, language=["pt"])
    lista_documentos = loader.load()
    documento = "\n\n".join([doc.page_content for doc in lista_documentos])
    return documento

# Interface Streamlit
st.title("Perguntas sobre Vídeos do YouTube")
url_input = st.text_input("Cole a URL do vídeo do YouTube:")
question = st.text_input("Digite sua pergunta sobre o vídeo:")

if st.button("Enviar"):
    if not url_input:
        st.error("Por favor, cole a URL do vídeo.")
    elif not question:
        st.error("Por favor, digite sua pergunta.")
    else:
        # Extrai o video_id da URL
        parsed = urlparse(url_input)
        video_ids = parse_qs(parsed.query).get("v")
        if not video_ids:
            st.error("URL inválida. Não foi possível extrair o ID do vídeo.")
        else:
            video_id = video_ids[0]
            with st.spinner("Carregando legenda do vídeo..."):
                texto = carrega_youtube(video_id)
            if not texto:
                st.error("Não foi possível carregar as legendas do vídeo.")
            else:
                with st.spinner("Processando resposta..."):
                    docs = [Document(page_content=texto)]
                    resposta = chain.run(input_documents=docs, question=question)
                st.subheader("Resposta")
                st.write(resposta)
