import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq
import os
import re
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
from datetime import datetime
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Annotated, TypedDict
import nltk
from nltk.corpus import stopwords
import datetime
import random
import uuid
import altair as alt
import os
from dotenv import load_dotenv
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
import json
import dateutil.parser
from funciones import get_answer_st
import plotly.express as px

# Cargar variables de entorno
load_dotenv()

# Configuración de la aplicación Streamlit
st.set_page_config(page_title="FifthWall Travel Planner",
                   layout='wide')
sentimientos = pd.read_csv("sentimientos.csv")
st.title("FifthWall Travel Planner🚞")
col1, col2 = st.tabs(["Chat", "Graph"])



# Cargar las variables de entorno desde el archivo .env
load_dotenv()



# Agrupar y contar los sentimientos
if 'Feeling' in sentimientos.columns:
    # Agrupar por el campo 'Feeling' y contar la cantidad de ocurrencias de cada sentimiento
    grupos = sentimientos.groupby("Feeling").size().reset_index(name='Count')

    # Mostrar el resultado en Streamlit
    with col2:

        # Crear un gráfico de tipo donut con Plotly
        fig = px.pie(
            grupos, 
            values='Count', 
            names='Feeling', 
            title='Distribución de Sentimientos',
            hole=0.4,  # Esto hace que sea un gráfico de anillo
            color_discrete_sequence=px.colors.qualitative.Pastel
        )

        # Mostrar el gráfico en Streamlit
        st.plotly_chart(fig, use_container_width=True)

else:
    st.error("La columna 'Feeling' no existe en la tabla de sentimientos.")




# Asegúrate de que conversation_id está inicializado
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())



# Estado de la sesión
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hola! ¿Como te sientes el dia de hoy?"),
    ]

# Estado de la sesión
if "chat_history" not in st.session_state:
    st.markdown("# Hola!👋🏻 como te sientes el dia de hoy?")

with col1:

    # Conversación
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
# Entrada del usuario
user_query = st.chat_input("Escriba acá sus intereses...")


def save_feeling_to_csv(chat_history, conversation_id, feeling):
    feeling_data = {
        "ConversationID": conversation_id,
        "Timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
        "Feeling": feeling
    }

    feeling_df = pd.DataFrame([feeling_data])
    feeling_file = "sentimientos.csv"
    
    # Cargar el archivo existente si existe
    if os.path.exists(feeling_file):
        existing_df = pd.read_csv(feeling_file)
        # Concatenar y eliminar duplicados
        updated_df = pd.concat([existing_df, feeling_df]).drop_duplicates(subset=['ConversationID', 'Timestamp', 'Feeling'], keep='last')
        updated_df.to_csv(feeling_file, index=False)
    else:
        feeling_df.to_csv(feeling_file, index=False)
    



# En la parte donde procesas la entrada del usuario
if user_query:
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):

        response = st.write_stream(get_answer_st(
            user_query[:500], 
            chat_history=str(st.session_state.chat_history[-5:])[:2000],
            uuid = st.session_state.conversation_id
        ))
        print(f'------{response}-------')
        st.session_state.chat_history.append(AIMessage(content=response))
        
