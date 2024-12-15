from flask import Flask, request
from twilio.rest import Client
import os
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime
from funciones import get_answer
# Cargar variables de entorno

load_dotenv()

# Init the Flask App
app = Flask(__name__)
# Twilio credentials (should be set as environment variables in a real application)
account_sid = os.getenv('TWILIO_ACCOUNT_SID', os.environ['TWILIO_ACCOUNT_SID'])
auth_token = os.getenv('TWILIO_AUTH_TOKEN', os.environ['TWILIO_AUTH_TOKEN'])
client = Client(account_sid, auth_token)
TWILIO_WHATSAPP_NUMBER = 'whatsapp:+14155238886'
# Cargar la base de cobranza
# Simple cache to store the step and credit for each user

cache_sac = {}
# Función para guardar las conversaciones en un CSV
def guardar_conversacion(numero, mensaje, origen):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    numero = str(numero).replace("whatsapp:+57", "")
    nuevo_registro = pd.DataFrame([[numero, mensaje, origen, timestamp]], columns=['Numero', 'Mensaje', 'Origen', 'Timestamp'])
    
    # Guardar en el archivo CSV
    nuevo_registro.to_csv('conversaciones.csv', mode='a', header=not os.path.exists('conversaciones.csv'), index=False, sep=";")


# Define a route to handle incoming requests
@app.route('/fifth', methods=['POST'])
def cobranza():
    incoming_msg = request.values.get('Body', '').lower()
    from_number = request.values.get('From', '')
    guardar_conversacion(from_number, incoming_msg, 'cliente')
    # Uso de un cache separado para el chat SAC
    cache_key_sac = f"{from_number}_sac"
    cache_sac.setdefault(cache_key_sac, {'context': []})
    
    # Procesa el mensaje del cliente en el contexto SAC
    cache_sac[cache_key_sac]['context'].append(f"user:{incoming_msg}")
    response_message = get_answer(incoming_msg, from_number,cache_sac[cache_key_sac]['context'])
    cache_sac[cache_key_sac]['context'].append(f"chatbot:{response_message}")
    guardar_conversacion(from_number, response_message, 'chatbot')
    client.messages.create(
        body=response_message,
        from_=TWILIO_WHATSAPP_NUMBER,
        to=from_number
    )
    return '', 204
# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port=5000)