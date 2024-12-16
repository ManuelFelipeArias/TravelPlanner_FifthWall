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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import nltk
import random

# Load environment variables
load_dotenv()
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))



def save_chat_message(conversation_id, user_message, bot_response):
    conn = sqlite3.connect('cache.db')
    c = conn.cursor()
    # Convert bot_response to string if it's not already
    if not isinstance(bot_response, str):
        bot_response = ''.join(bot_response) if hasattr(bot_response, '__iter__') else str(bot_response)

    # Insert the chat message and automatically capture ROWID and timestamp
    c.execute(
        """
        INSERT INTO chat_history (conversation_id, user_message, bot_response) 
        VALUES (?, ?, ?)
        """,
        (conversation_id, user_message, bot_response)
    )
    conn.commit()
    # Close the connection
    conn.close()


def save_data_travel(json, uuid):
    conn = sqlite3.connect('cache.db')
    c = conn.cursor()
    # Extract values from the JSON input
    summary = json.get("summary")
    dest = json.get("dest")
    budget = json.get("budget")
    days = json.get("days")
    
    # Connect to SQLite database
    conn = sqlite3.connect('cache.db')
    c = conn.cursor()
    
    # Insert data into the travel_user_data table
    c.execute(
        """
        INSERT INTO travel_user_data (uuid, summary, destination, budget, days)
        VALUES (?, ?, ?, ?, ?)
        """,
        (uuid, summary, dest, budget, days)
    )
    
    # Commit the transaction
    conn.commit()
    
    # Close the connection
    conn.close()


# Preprocess text
def text_preprocess(text):
    text = str(text).lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r',', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words).strip()

# Split text into chunks
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

# Load initial data
df_train = pd.read_csv("hf://datasets/osunlp/TravelPlanner/train.csv")
df_train['Texto'] = df_train['reference_information'].dropna().apply(lambda x: re.sub(r'\s+', ' ', str(x)).strip())

chunks_list = []
for text in df_train['Texto']:
    chunks = text_to_chunks(text)
    chunks_list.extend(chunks)

preprocessed_texts = [text_preprocess(chunk) for chunk in chunks_list]

# Create TF-IDF matrix
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_texts)


def get_travel_data(chat_history):
    template = """
You are a segmentation model for chat history analysis. Your task is to analyze each message and extract relevant information about **summary**,**org**, **dest**,**budget**, and **time**, returning the results in a structured JSON format.


### Only possible Output:
```
    "summary": "I want to travel to Paris from New York in December 3 days, I have $1500 for the trip.",
    "org": "New York",
    "dest": "Paris",
    "budget": 1500,
    "days": "3"
```

    ### Chat History:
    {chat_history}
One json per request. NOT dict, just json.
    ### Expected Output:

    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGroq(groq_api_key=os.environ['GROQ_API_KEY'], model_name="llama-3.3-70b-versatile")
    chain = prompt | llm | JsonOutputParser()

    ai_query = chain.invoke({"chat_history": chat_history[:5000]}) # Limit history to prevent API issues.
    return ai_query


# Get response based on user query and chat history
def get_response(user_query, chat_history, relevant_docs):
    template = """
You are Charlie a travel planner and agent for FifthWall Solutions travel insurance. Help users find their perfect trip and recommend travel insurance. Use {relevant_docs} for trip details.

Instructions:
Plan the Trip ✈️:

Ask for the summary, destination, budget, and time.
Suggest trips based on their preferences with details on accommodations, flights, activities, and costs.
Provide clear options with booking links.
Engage with the User 💬:

Ask follow-up questions to refine details.
Offer multiple options and highlight benefits.
Travel Insurance 🛡️:

Once they decide, recommend FifthWall Solutions Travel Insurance.
Explain coverage (cancellations, medical emergencies, etc.).
Link to purchase: FifthWall Solutions Travel Insurance.
User input:{user_query}
very short answers
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGroq(groq_api_key=os.environ.get('GROQ_API_KEY'), model_name="llama-3.3-70b-versatile")

    try:
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({
            "chat_history": chat_history,
            "user_query": user_query,
            "relevant_docs": relevant_docs
        })
        print(response)
        return response
    except Exception as e:
        return "Error"
    

def get_response_st(user_query, chat_history, relevant_docs):
    template = """
You are Charlie a travel planner and agent for FifthWall Solutions travel insurance. Help users find their perfect trip and recommend travel insurance. Use {relevant_docs} for trip details.

Instructions:
Plan the Trip ✈️:

Ask for the summary, destination, budget, and time.
Suggest trips based on their preferences with details on accommodations, flights, activities, and costs.
Provide clear options with booking links.
Engage with the User 💬:

Ask follow-up questions to refine details.
Offer multiple options and highlight benefits.
Travel Insurance 🛡️:

Once they decide, recommend FifthWall Solutions Travel Insurance.
Explain coverage (cancellations, medical emergencies, etc.).
Link to purchase: FifthWall Solutions Travel Insurance.
User input:{user_query}
very short answers
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGroq(groq_api_key=os.environ.get('GROQ_API_KEY'), model_name="llama-3.3-70b-versatile")

    try:
        chain = prompt | llm | StrOutputParser()
        response = chain.stream({
            "chat_history": chat_history,
            "user_query": user_query,
            "relevant_docs": relevant_docs
        })
        print(response)
        return response
    except Exception as e:
        return "Error"
    

# Main function to process user queries
def get_answer(user_query, uuid, chat_history):
    query_processed = text_preprocess(user_query[:500])

    reference_value_vector = tfidf_vectorizer.transform([query_processed])

    similarities = cosine_similarity(reference_value_vector, tfidf_matrix)[0]
    top_2_indices = similarities.argsort()[-2:][::-1]
    
    relevant_docs = [chunks_list[i] for i in top_2_indices]

    response = get_response(
        user_query[:500], 
        chat_history,
        relevant_docs=str(relevant_docs)
    )
    print(f"---------------------------{len(chat_history)}-------------------------------------")
    save_chat_message(bot_response=str(response),conversation_id=uuid,user_message=user_query)
    try:
        if len(chat_history) ==3 or len(chat_history) == 9 or len(chat_history) == 15:
            json_data = get_travel_data(chat_history)
            print(json_data)
            save_data_travel(json_data,uuid=uuid)
    except:
        pass
    return response


# Main function to process user queries
def get_answer_st(user_query, uuid, chat_history,conv_len):
    query_processed = text_preprocess(user_query[:500])

    reference_value_vector = tfidf_vectorizer.transform([query_processed])

    similarities = cosine_similarity(reference_value_vector, tfidf_matrix)[0]
    top_2_indices = similarities.argsort()[-2:][::-1]
    
    relevant_docs = [chunks_list[i] for i in top_2_indices]

    response = get_response_st(
        user_query[:500], 
        chat_history,
        relevant_docs=str(relevant_docs)
    )
    save_chat_message(bot_response=str(response),conversation_id=uuid,user_message=user_query)
    try:
        if conv_len== 4 or conv_len== 8 or conv_len== 15:
            json_data = get_travel_data(chat_history)
            print(json_data)
            save_data_travel(json_data,uuid=uuid)
    except:
        pass
    return response
