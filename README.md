# Documentation: Chatbot Travel Planner System

This project contains three main Python modules that form a travel chatbot system. The application processes user inputs, manages chat history, recommends trips, and integrates travel insurance offerings. Below is the documentation for each module and its functionality.

---

## **Module: `funciones.py`**

This module contains utility functions for text preprocessing, data management, and chatbot responses. 

### **1. Key Libraries and Initialization**
- **Imports**: Includes `sqlite3`, `langchain_groq`, `nltk`, `pandas`, `sklearn`, and other essential libraries for processing text, managing data, and interfacing with LLMs.
- **Environment Setup**: 
  - Loads environment variables with `dotenv`.
  - Downloads NLTK stopwords.
  - Initializes a stop words list.

---

### **2. Key Functions**

#### **2.1 Chat History Management**
- **`save_chat_message(conversation_id, user_message, bot_response)`**
  - Saves a chat message to an SQLite database (`cache.db`).
  - Handles `bot_response` type to ensure it is stored as a string.

- **`save_data_travel(json, uuid)`**
  - Stores extracted travel details (`summary`, `destination`, `budget`, `days`) from JSON into an SQLite database.

---

#### **2.2 Text Processing**
- **`text_preprocess(text)`**
  - Cleans and tokenizes input text, removing stop words and special characters.

- **`text_to_chunks(text, chunk_size=500)`**
  - Splits large text blocks into smaller chunks of a specified size.

---

#### **2.3 TF-IDF Vectorizer for Similarity**
- **`get_answer(user_query, uuid, chat_history)`**
  - Preprocesses user queries.
  - Calculates text similarity with preloaded TF-IDF matrix.
  - Fetches relevant documents to enhance chatbot responses.

- **`get_answer_st(user_query, uuid, chat_history, conv_len)`**
  - Similar to `get_answer`, but streams the response for dynamic interaction.

---

#### **2.4 LLM Integration**
- **`get_travel_data(chat_history)`**
  - Uses a prompt template to extract travel details from chat history.
  - Returns structured JSON with extracted travel information.

- **`get_response(user_query, chat_history, relevant_docs)`**
  - Generates a response based on user queries, chat history, and relevant documents using Groq API.

---

---

## **Module: `twilio_app.py`**

This module integrates the chatbot with Twilio's WhatsApp API using Flask.

### **1. Key Libraries and Initialization**
- **Imports**: Includes `flask`, `twilio`, `dotenv`, and `funciones.py`.
- **Environment Setup**: Loads Twilio credentials (`TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`) and initializes the Flask app.

---

### **2. Chat Message Processing**
- **`guardar_conversacion(numero, mensaje, origen)`**
  - Saves conversations to a CSV (not implemented in this version).

- **Endpoint: `/fifth`**
  - Receives WhatsApp messages via POST.
  - Processes user queries using `get_answer`.
  - Sends a response back via Twilio's API.

---

### **3. Twilio Integration**
- **WhatsApp Messaging**
  - `TWILIO_WHATSAPP_NUMBER`: Default sender number.
  - Sends replies directly to the user through Twilio's WhatsApp service.

---

---

## **Module: `streamlit_rag.py`**

This module provides a Streamlit interface for the chatbot.

### **1. Key Libraries and Initialization**
- **Imports**: Includes `streamlit`, `langchain_core`, `dotenv`, `uuid`, `plotly.express`, and `funciones.py`.
- **Environment Setup**: Loads environment variables.

---

### **2. Streamlit App Configuration**
- Streamlit serves as the user interface for:
  - Inputting queries.
  - Viewing chatbot responses.
  - Interacting with travel recommendations.

---

## **Database Schema**
### **1. SQLite Tables**
- **`chat_history`**
  - `conversation_id`: Unique identifier for the conversation.
  - `user_message`: User's input message.
  - `bot_response`: Chatbot's reply.

- **`travel_user_data`**
  - `uuid`: Unique identifier for travel requests.
  - `summary`, `destination`, `budget`, `days`: Extracted travel details.

---

## **Project Workflow**
1. **Input Processing**:
   - Users send queries via WhatsApp or Streamlit.
   - Text is preprocessed and matched with relevant training data using TF-IDF similarity.

2. **Response Generation**:
   - Relevant documents are used to generate coherent chatbot replies.
   - Travel details are extracted and saved when specific conditions are met.

3. **Output Delivery**:
   - WhatsApp responses are sent through Twilio.
   - Streamlit displays output for direct user interaction.

4. **Data Storage**:
   - Chat history and travel details are logged in the SQLite database for future reference.

---

## **How to Run the Project**
1. **Setup Environment**:
   - Install required libraries using `pip install -r requirements.txt`.
   - Configure environment variables in a `.env` file:
     ```
     GROQ_API_KEY=your_groq_api_key
     TWILIO_ACCOUNT_SID=your_twilio_account_sid
     TWILIO_AUTH_TOKEN=your_twilio_auth_token
     ```

2. **Run Flask App**:
   - Start the WhatsApp integration with:
     ```bash
     python twilio_app.py
     ```

3. **Run Streamlit App**:
   - Launch the Streamlit interface with:
     ```bash
     streamlit run streamlit_rag.py
     ```

4. **Database Initialization**:
   - Ensure `cache.db` SQLite database exists with appropriate schemas.

---

## **Future Enhancements**
- Improve error handling for API and database interactions.
- Extend travel data extraction to include more complex scenarios.
- Enhance the Streamlit interface for a better user experience.
- Optimize TF-IDF processing for large datasets.

---
