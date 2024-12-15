import sqlite3
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
import random
import uuid
import json
import dateutil.parser

# Cargar variables de entorno
load_dotenv()
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# # Configurar base de datos SQLite
# conn = sqlite3.connect('cache.db')
# c = conn.cursor()

# # Crear tablas necesarias
# c.execute('''
# CREATE TABLE IF NOT EXISTS sentimientos (
#     id INTEGER PRIMARY KEY AUTOINCREMENT,
#     feeling TEXT,
#     timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
# )
# ''')
# c.execute('''
# CREATE TABLE IF NOT EXISTS chat_history (
#     id INTEGER PRIMARY KEY AUTOINCREMENT,
#     conversation_id TEXT,
#     user_message TEXT,
#     bot_response TEXT,
#     timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
# )
# ''')
# conn.commit()

# # Función para almacenar sentimientos
# def save_feeling(feeling):
#     c.execute("INSERT INTO sentimientos (feeling) VALUES (?)", (feeling,))
#     conn.commit()

# # Función para obtener el historial de sentimientos
# def get_feelings():
#     c.execute("SELECT feeling, COUNT(*) as count FROM sentimientos GROUP BY feeling")
#     return c.fetchall()

# # Función para guardar un mensaje en el historial
# def save_chat_message(conversation_id, user_message, bot_response):
#     c.execute(
#         "INSERT INTO chat_history (conversation_id, user_message, bot_response) VALUES (?, ?, ?)",
#         (conversation_id, user_message, bot_response)
#     )
#     conn.commit()

# # Función para recuperar historial de una conversación
# def get_chat_history(conversation_id):
#     c.execute("SELECT user_message, bot_response FROM chat_history WHERE conversation_id = ?", (conversation_id,))
#     return c.fetchall()

# Cargar datos iniciales
df_train = pd.read_csv("hf://datasets/osunlp/TravelPlanner/train.csv")
df_train['Texto'] = df_train['reference_information'].dropna().apply(lambda x: re.sub(r'\s+', ' ', str(x)).strip())

# Procesar texto
def text_preprocess(text):
    text = str(text).lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r',', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words).strip()

def text_to_chunks(text, chunk_size=500):
    words = text.split()
    chunks = []
    current_chunk = ''
    word_count = 0
    for word in words:
        current_chunk += word + ' '
        word_count += 1
        if word_count >= chunk_size:
            chunks.append(current_chunk.strip())
            current_chunk = ''
            word_count = 0
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

chunks_list = []
for text in df_train['Texto']:
    chunks = text_to_chunks(text)
    chunks_list.extend(chunks)

preprocessed_texts = [text_preprocess(chunk) for chunk in chunks_list]

# Crear matriz TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_texts)

# Función para generar ID de radicado
def generate_radicado():
    current_date = datetime.now()
    date_string = current_date.strftime("%Y%m%d")
    random_number = random.randint(1000, 9999)
    return f"RAD-{date_string}-{random_number}"

# Función de segmentación de sentimientos
def get_feeling(user_query, chat_history):
    template = """
    Eres un segmentador de sentimientos para el siguiente chat. Selecciona el sentimiento que el usuario tiene:

    Historia del chat: `{chat_history}`   
    Pregunta del usuario: `{user_query}`

    Las opciones para "feeling" son:
    - Feliz
    - Tranquilo
    - Animado
    - Amado
    - Enojado
    - Angustiado
    - Estresado
    - Deprimido

    Responde con un JSON bien formado:
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGroq(groq_api_key=os.environ['GROQ_API_KEY'], model_name="llama-3.1-70b-versatile")
    chain = prompt | llm | JsonOutputParser()

    ai_query = chain.invoke({
        "user_query": user_query, 
        "chat_history": chat_history[:3000]
    })
    return ai_query

def get_response_st(user_query, chat_history, relevant_docs):
    
    template = """
    Documentos relevantes: {relevant_docs}
    Eres el mejor agente Pscicologo del mundo y el contexto te va a ayudar a dar mejores respuestas con respecto a lo que las personas sienten. Siempre respondes con la intención de que el usuario se sienta mejor y pueda resolver sus problemas de sentirse mal a bien ese es tu objetivo, puedes usar emojis.:
    Chat histórico: {chat_history}
    Pregunta del usuario: {user_query}
    Sé breve en la respuesta.
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGroq(groq_api_key=os.environ.get('GROQ_API_KEY'), model_name="llama-3.1-70b-versatile")

    try:
        chain = prompt | llm | StrOutputParser()
        respuesta = chain.stream({
              "chat_history": chat_history,
              "user_query": user_query,
              "relevant_docs": relevant_docs
          })
        print(respuesta)
        return respuesta
    except Exception as e:
        return "Error"

def get_response(user_query, chat_history, relevant_docs):
    
    template = """
    Documentos relevantes: {relevant_docs}
    Eres el mejor agente Pscicologo del mundo y el contexto te va a ayudar a dar mejores respuestas con respecto a lo que las personas sienten. Siempre respondes con la intención de que el usuario se sienta mejor y pueda resolver sus problemas de sentirse mal a bien ese es tu objetivo, puedes usar emojis.:
    Chat histórico: {chat_history}
    Pregunta del usuario: {user_query}
    Sé breve en la respuesta.
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGroq(groq_api_key=os.environ.get('GROQ_API_KEY'), model_name="llama-3.1-70b-versatile")

    try:
        chain = prompt | llm | StrOutputParser()
        respuesta = chain.invoke({
              "chat_history": chat_history,
              "user_query": user_query,
              "relevant_docs": relevant_docs
          })
        print(respuesta)
        return respuesta
    except Exception as e:
        return "Error"

def get_answer_st(user_query, uuid,chat_history):
    query_processed = text_preprocess(user_query[:500])

    # Usa transform en lugar de fit_transform aquí
    reference_value_vector = tfidf_vectorizer.transform([query_processed])

    similarities = cosine_similarity(reference_value_vector, tfidf_matrix)[0]
    # Obtener los 2 documentos más relevantes
    top_2_indices = similarities.argsort()[-2:][::-1]
    
    relevant_docs = [chunks_list[i] for i in top_2_indices]

    response = (get_response_st(
            user_query[:500], 
            chat_history,
            relevant_docs=str(relevant_docs)
        ))
    print(f'------{relevant_docs}-------')
    
    # Guardar sentimiento si el historial tiene suficiente contenido
    # chat_history = get_chat_history(uuid)
    if len(chat_history) >= 5:
        try:
            feeling = get_feeling(user_query, chat_history)
            ai_feeling = feeling["feeling"]
            # save_feeling(ai_feeling)
        except Exception as e:
            print(f"Error al determinar sentimiento: {e}")

    return response

def get_answer(user_query, uuid,chat_history):
    query_processed = text_preprocess(user_query[:500])

    # Usa transform en lugar de fit_transform aquí
    reference_value_vector = tfidf_vectorizer.transform([query_processed])

    similarities = cosine_similarity(reference_value_vector, tfidf_matrix)[0]
    # Obtener los 2 documentos más relevantes
    top_2_indices = similarities.argsort()[-2:][::-1]
    
    relevant_docs = [chunks_list[i] for i in top_2_indices]

    response = (get_response(
            user_query[:500], 
            chat_history,
            relevant_docs=str(relevant_docs)
        ))
    print(f'------{relevant_docs}-------')
    
    # Guardar sentimiento si el historial tiene suficiente contenido
    # chat_history = get_chat_history(uuid)
    if len(chat_history) >= 5:
        try:
            feeling = get_feeling(user_query, chat_history)
            ai_feeling = feeling["feeling"]
            # save_feeling(ai_feeling)
        except Exception as e:
            print(f"Error al determinar sentimiento: {e}")

    return response
