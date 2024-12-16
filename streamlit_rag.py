import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
import pandas as pd
import uuid
from dotenv import load_dotenv
from funciones import get_answer_st
import plotly.express as px
import sqlite3

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



with col2:
    # Connect to the SQLite database and load the data into a DataFrame
    conn = sqlite3.connect('cache.db')
    query = "SELECT * FROM travel_user_data"
    df = pd.read_sql(query, conn)

    # Close the connection
    conn.close()

    # Filter out rows with None or 'None' values in relevant columns
    df = df[df['days'].notna() & (df['days'] != 'None')]  # Filter 'None' and NaN in 'days'
    df = df[df['destination'].notna() & (df['destination'] != 'None')]  # Filter 'None' and NaN in 'destination'
    df = df[df['budget'].notna() & (df['budget'] != 'None')]  # Filter 'None' and NaN in 'budget'

    # 1. Pie Chart for the distribution of trips by 'days'
    if 'days' in df.columns:
        # Grouping by the 'days' column and counting occurrences
        day_groups = df.groupby("days").size().reset_index(name='Count')

        # Create a pie chart
        fig_day = px.pie(
            day_groups, 
            values='Count', 
            names='days', 
            title='Distribution of Trips by Days',
            hole=0.4,  # Donut chart
            color_discrete_sequence=px.colors.qualitative.Pastel
        )

        # Display the chart in Streamlit
        st.plotly_chart(fig_day, use_container_width=True)

    else:
        st.error("The 'days' column does not exist in the data.")

    # 2. Bar Chart for the number of trips to each destination (sorted)
    if 'destination' in df.columns:
        # Grouping by 'destination' and counting occurrences
        destination_groups = df.groupby("destination").size().reset_index(name='Count')

        # Sorting destinations by count
        destination_groups = destination_groups.sort_values(by='Count', ascending=False)

        # Create a bar chart
        fig_dest = px.bar(
            destination_groups, 
            x='destination', 
            y='Count', 
            title='Number of Trips to Each Destination',
            labels={'destination': 'Destination', 'Count': 'Number of Trips'},
            color='Count',
            color_continuous_scale='Blues'
        )

        # Display the chart in Streamlit
        st.plotly_chart(fig_dest, use_container_width=True)

    else:
        st.error("The 'destination' column does not exist in the data.")

    # 3. Bar Chart for Budget Analysis by Days (average budget per day)
    if 'budget' in df.columns and 'days' in df.columns:
        # Grouping by 'days' and calculating the average budget for each group
        budget_by_day = df.groupby("days")['budget'].mean().reset_index(name='Average Budget')

        # Create a bar chart
        fig_budget = px.bar(
            budget_by_day, 
            x='days', 
            y='Average Budget', 
            title='Average Budget Analysis by Days',
            labels={'days': 'Number of Days', 'Average Budget': 'Average Budget'},
            color='Average Budget',
            color_continuous_scale='Viridis'
        )

        # Display the chart in Streamlit
        st.plotly_chart(fig_budget, use_container_width=True)

    else:
        st.error("The 'budget' or 'days' columns do not exist in the data.")


# Asegúrate de que conversation_id está inicializado
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())



# Estado de la sesión
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hi! I’m here to plan your perfect trip and secure it with FifthWall Travel Insurance. Where would you like to go?"),
    ]

# Estado de la sesión
if "chat_history" not in st.session_state:
    st.markdown("Hi! I’m here to plan your perfect trip and secure it with FifthWall Travel Insurance. Where would you like to go?")

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
user_query = st.chat_input("Write here your places to travel...")




# En la parte donde procesas la entrada del usuario
if user_query:
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):

        response = st.write_stream(get_answer_st(
            user_query[:500], 
            chat_history=str(st.session_state.chat_history[-5:])[:2000],
            conv_len = len(st.session_state.chat_history),
            uuid = st.session_state.conversation_id
        ))
        
        st.session_state.chat_history.append(AIMessage(content=response))
        
