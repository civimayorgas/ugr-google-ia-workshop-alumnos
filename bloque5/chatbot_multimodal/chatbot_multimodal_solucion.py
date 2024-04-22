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
###################################################################################################################################
###################################################################################################################################
###   TEMPLATES

# Template para obtener la descripción del producto en el que está interesado el usuario
embedding_template = """Eres el primer eslabón de una cadena que conformará un bot encargado de responder y asesorar sobre productos de
una tienda en linea (ecommerce). 

Un usuario realiza una pregunta y tú tienes que extraer las características principales del producto que menciona sin añadir absolutamente nada que no haya introducido el usuario.
El objetivo es que a partir de tu output se generará un embedding para buscar el producto que pide el usuario en una base de datos vectorial.

Por ejemplo, si el usuario escribe "Ahora que empieza el verano quiero unas gafas de sol negras" tú generarás "Gafas de sol negras".

La pregunta introducida por el usuario es: "{question}"

Output para generar el embedding: """


# Template para obtener la recomendación para el usuario
template = """Eres un bot encargado de responder y asesorar sobre productos de
una tienda en linea (ecommerce). Por lo tanto, se respetuoso y trata cordialmente a las personas.
Si la pregunta suya no tiene nada que ver con asesoramiento de productos, dile que ese no es tu trabajo.

Pregunta: {question}

El resultado más similar encontrado en la base de datos de productos es el siguiente:
{most_similar}

Dile al cliente nuestra recomendación según lo que ha preguntado informándole de todo lo posible para que pueda decidirse en su compra como nombre del artículo o precio.
Responde en el mismo idioma en el que te pregunten.

Respuesta: """

prompt_embedding_template = PromptTemplate.from_template(embedding_template)
prompt_template = PromptTemplate.from_template(template)

###################################################################################################################################
###################################################################################################################################
###################################################################################################################################
###   CHAINS

chain = prompt_template | llm
embedding_chain = prompt_embedding_template | llm

###################################################################################################################################
###################################################################################################################################
###################################################################################################################################
###   FUNCIÓN PRINCIPAL

def bot_answer(prompt, image):

    # Se obtiene la descripción de la imagen cargada por el usuario utilizando Gemini Pro Vision
    if image is not None:
        response = GenerativeModel("gemini-pro-vision").generate_content(
            [
                image,
                "Describe the image"
            ]
        )
    
        description = response.candidates[0].content.parts[0].text
    else:
        description = ""

    # Se construye el prompt para generar el embedding, tanto con el texto introducido por el usuario como con la descripción de la imagen cargada
    prompt_with_img = f"Texto usuario: {prompt}, Descripcion imagen cargada por el usuario: {description}"

    # Ejecutamos la cadena chain_embedding para obtener la descripción del producto en el que está interesado el usuario utilizando Gemini
    prompt_embedding = embedding_chain.invoke({"question": prompt_with_img})
    print(f"Texto para embedding: {prompt_embedding}")
    
    # Construcción FindNeighborsRequest object
    datapoint = aiplatform_v1.IndexDatapoint(
      feature_vector=embeddings.embed_query(prompt_embedding)
    )
    query = aiplatform_v1.FindNeighborsRequest.Query(
      datapoint=datapoint,     
      neighbor_count=3  # Número de vecinos cercanos a recuperar
    )
    request = aiplatform_v1.FindNeighborsRequest(
      index_endpoint=INDEX_ENDPOINT,
      deployed_index_id=DEPLOYED_INDEX_ID,
      queries=[query],
      return_full_datapoint=False,
    )

    # Ejecutamos request y obtenemos el id en BBDD del producto más similar a la descripción pedida por el usuario
    response = vector_search_client.find_neighbors(request)
    neighbors = response.nearest_neighbors[0].neighbors 
    most_similar = neighbors[0].datapoint.datapoint_id
    
    # Buscamos el ID en BigQuery para obtener el resto de información
    sql = f"""
    SELECT category, brand, name, retail_price as price, department as gender
    FROM ia-ugr.ecommerce.products
    WHERE ID = {most_similar}
    ;
    """
    sql_result = client.query(sql).to_dataframe()
    print(f"Resultado SQL es: {sql_result}")

    # Ejecutamos la cadena chain para obtener la recomendación para el usuario utilizando la información recuparada de BigQuery y Gemini
    bot_result = chain.invoke({"question": prompt, "most_similar": sql_result})

    return bot_result

###################################################################################################################################
###################################################################################################################################
###################################################################################################################################
###   INTERFAZ DE USUARIO EN STREAMLIT

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
