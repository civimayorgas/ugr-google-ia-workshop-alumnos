from google.cloud import aiplatform_v1
import streamlit as st 
from google.cloud import bigquery
from langchain_google_vertexai.llms import VertexAI
from langchain_core.prompts import PromptTemplate
import vertexai
from vertexai.preview.generative_models import (
    GenerativeModel,
    Part,
    Image,
)
import os
from langchain_google_vertexai import VertexAIEmbeddings

# Configuración para el índice desplegado
API_ENDPOINT="2069128544.us-central1-1043238928011.vdb.vertexai.goog"
INDEX_ENDPOINT="projects/1043238928011/locations/us-central1/indexEndpoints/6459098649556156416"
DEPLOYED_INDEX_ID="products_data_index_civica"

# Cliente Vector Search
client_options = {"api_endpoint": API_ENDPOINT}
vector_search_client = aiplatform_v1.MatchServiceClient(client_options=client_options)

# Cliente BigQuery
client = bigquery.Client()

# Obtención embeddings
embeddings = VertexAIEmbeddings(model_name="textembedding-gecko@001")

# Configuración LLM (Gemini)
llm = VertexAI(model_name="gemini-pro", temperature=0.1)


###################################################################################################################################
# ##################################################################################################################################
# ##################################################################################################################################
# ##   TEMPLATES

# Template para obtener la descripción del producto en el que está interesado el usuario
embedding_template = """ ... """


# Template para obtener la recomendación para el usuario
template = """ ... """

prompt_embedding_template = PromptTemplate.from_template(embedding_template)
prompt_template = PromptTemplate.from_template(template)

###################################################################################################################################
# ##################################################################################################################################
# ##################################################################################################################################
# ##   CHAINS

chain = 
embedding_chain = 

###################################################################################################################################
# ##################################################################################################################################
# ##################################################################################################################################
# ##   FUNCIÓN PRINCIPAL

def bot_answer(prompt, image):

    return bot_result

###################################################################################################################################
# ##################################################################################################################################
# ##################################################################################################################################
# ##   INTERFAZ DE USUARIO EN STREAMLIT

# Título
st.title("Chatbot de ecommerce :sunglasses:")
st.markdown("""
Pregunta lo que quieras para ser asesorado sobre los productos que hay en nuestra tienda. ¡Prueba a cargar una imagen!
""")

# Inicializamos mensajes
INITIAL_MESSAGE = [
    {
        "role": "assistant",
        "content": "¡Hola!, soy tu asistente de compras. Pregúntame lo que quieras y estaré encantado de buscar entre todo el almacén de productos lo mejor para ti.",
    },
]
if "messages" not in st.session_state.keys():
    st.session_state["messages"] = INITIAL_MESSAGE

if "history" not in st.session_state:
    st.session_state["history"] = []
    
# Muestra la historia de mensajes on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Se añade funcionalidad para cargar imágenes
if uploaded_file := st.file_uploader("Carga una imagen"):
    image = Image.from_bytes(uploaded_file.read())
else:
    image = None

# input del usuario
if prompt := st.chat_input("¿Qué quieres saber?"):

    # Añade el mensaje del usuario a la historia de mensajes
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Muestra el mensaje del usuario
    with st.chat_message("user"):
        st.markdown(prompt)

    # Muestra el mensaje del asistente
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = bot_answer(prompt, image)  # Respuesta del chatbot
        message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})
