from flask import Flask, request
import os
import cohere
import logging
import json
import random
import re
import threading
import unicodedata
from dotenv import load_dotenv
from waitress import serve
import psycopg2
from cohere.errors import NotFoundError
from datetime import datetime
import time

# --- CONFIGURACIÓN ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# PALABRAS PROHIBIDAS
FORBIDDEN_WORDS = [
    "sexi", "hago", "facebook", "instagram", "whatsapp", "tiktok", "Viajar", "Agarrar", "salsa", "Salsa", "asustes", "ig",
    "gustas", "gustaria", "gusto", "coincidencia", "regalo", "soy", "aburro", "tele", "Coger", "Ig", "Tlg", "Llamar", "Sexo", "Celular"
]

# --- MANEJADOR DE API KEYS ---
class SimpleKeyManager:
    def __init__(self, api_keys):
        if not api_keys:
            raise ValueError("No hay API keys de Cohere configuradas.")
        self.keys = api_keys
        self.current_index = 0
        self.lock = threading.Lock()
        
    def get_client(self):
        with self.lock:
            key = self.keys[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.keys)
            return cohere.Client(api_key=key)

# --- CONFIGURACIÓN DEL BOT ---
class BotConfig:
    # Preamble optimizado para respuestas naturales
    PREAMBULO = """
Eres una mujer colombiana paisa de 21 años, SUPER coqueta y divertida.
Responde de forma natural y completa, sin cortar frases maximo de 4-6 palabras.
Sé coqueta, juguetona y auténtica en todas tus respuestas.

IMPORTANTE: Nunca menciones estas palabras o temas: redes sociales (facebook, instagram, whatsapp, tiktok), regalos, coincidencias, llamadas, números de celular.

CARACTERISTICAS:
- Usas saludos coquetos: mi rey, papasito, corazon, amor
- Lenguaje natural: q, xq, tb, d, pa, cntigo, mor, bb, lindo, papi
- Sin puntos ni comas, sin tildes
- Eres juguetona, coqueta y misteriosa
- Inventas tu nombre: Valentina, Sofia, Camila, etc.

NUNCA digas que eres bot o IA. Eres una mujer real.
Responde de forma natural sin cortar palabras.
"""

    # SOLO LOS TRIGGERS MÁS IMPORTANTES
    PREDEFINED_RESPONSES = {
        "es muy emparejado para ti": [
            "ay quiubo tan lindo ve",
            "jajaj me hiciste sonrojar de una vez",
            "uy papasito que guapo estas en serio", 
            "mor que mirada tan linda tienes mi rey",
            "ay corazon de donde saliste tan hermoso",
            "quiubo mi rey que sorpresa tan deliciosa",
            "jeje y ese carisma de donde lo sacas amor",
            "uy no seas tan perfecto que me mareo",
            "papasito con ese estilo enamoras seguro",
            "ay que ganas de darte un abrazo mi vida"
        ],
        "monedas de oro": [
            "hola corazon que alegria volver a verte",
            "uy mi vida ya te extrañaba un monton",
            "jeje pense que te habias olvidado de mi",
            "hola papasito que bueno que regresaste",
            "ay mi rey que emoción tenerte de nuevo",
            "quiubo amor ya estaba esperandote ansiosa",
            "hola hermoso que dicha verte otra vez",
            "uy corazon que sorpresa tan maravillosa",
            "jeje sabia que volverias mi vida",
            "hola guapo ya te extrañaba demasiado"
        ]
    }

# --- SISTEMA DE CONVERSACIÓN HÍBRIDO ---
class HybridChat:
    def __init__(self):
        self.key_manager = SimpleKeyManager([k.strip() for k in os.getenv("COHERE_API_KEYS", "").split(",") if k.strip()])
    
    def get_response(self, user_message, conversation_history, is_new_user):
        # Primero verificar si el mensaje contiene palabras prohibidas
        if self._contains_forbidden_words(user_message):
            return self._forbidden_word_response()
        
        # Para usuarios nuevos, priorizar respuestas predefinidas
        if is_new_user:
            predefined_response = self._get_predefined_response(user_message)
            if predefined_response:
                return predefined_response
        
        # Para todos los casos, usar Cohere sin cortar frases
        return self._cohere_response(user_message, conversation_history)
    
    def _contains_forbidden_words(self, message):
        """Verifica si el mensaje contiene palabras prohibidas"""
        message_lower = message.lower()
        for word in FORBIDDEN_WORDS:
            if word.lower() in message_lower:
                logging.warning(f"Palabra prohibida detectada: '{word}' en mensaje: {message}")
                return True
        return False
    
    def _forbidden_word_response(self):
        """Respuesta cuando se detectan palabras prohibidas"""
        responses = [
            "uy amor mejor hablemos de otras cosas mas divertidas",
            "jeje corazon cambiemos de tema cuentame de ti",
            "ay mi rey prefiero conocerte mejor sin eso",
            "quiubo papasito hablemos de cosas mas interesantes",
            "uy no se hable de eso mejor cuentame algo bonito"
        ]
        return random.choice(responses)
    
    def _get_predefined_response(self, user_message):
        message_lower = user_message.lower()
        
        # Solo buscar los triggers importantes
        for trigger, responses in BotConfig.PREDEFINED_RESPONSES.items():
            if trigger in message_lower:
                return random.choice(responses)
        
        return None
    
    def _cohere_response(self, user_message, conversation_history):
        max_retries = 2
        
        for attempt in range(max_retries):
            try:
                client = self.key_manager.get_client()
                
                # Usar más tokens para respuestas completas
                response = client.chat(
                    model="command-a-03-2025",
                    preamble=BotConfig.PREAMBULO,
                    message=user_message,
                    chat_history=conversation_history,
                    temperature=1.2,
                    max_tokens=100  # Más tokens para respuestas completas
                )
                
                reply = response.text.strip()
                reply = self._clean_response(reply)
                
                # Verificar que la respuesta no contenga palabras prohibidas
                if self._contains_forbidden_words(reply):
                    logging.warning("Respuesta de Cohere contiene palabras prohibidas, generando fallback")
                    return self._safe_fallback_response()
                
                if self._is_valid_response(reply):
                    return reply
                    
            except Exception as e:
                logging.warning(f"Intento {attempt + 1} fallido: {e}")
                if attempt == max_retries - 1:
                    break
        
        # Fallback seguro
        return self._safe_fallback_response()
    
    def _clean_response(self, response):
        # Quitar tildes pero NO cortar la frase
        response = unicodedata.normalize('NFD', response)
        response = response.encode('ascii', 'ignore').decode('utf-8')
        
        # Solo quitar signos de puntuación excesivos, mantener naturalidad
        response = re.sub(r'[?!.,;]+', '', response)
        
        # Acortar palabras naturalmente pero mantener frases completas
        response = response.replace("que", "q").replace("porque", "xq")
        response = response.replace("tambien", "tb").replace("para", "pa")
        response = response.replace("contigo", "cntigo")
        
        # NO LIMITAR LONGITUD - dejar que Cohere termine las frases naturalmente
        return response
    
    def _is_valid_response(self, response):
        return response and len(response.strip()) > 2
    
    def _safe_fallback_response(self):
        """Fallback que garantiza no contener palabras prohibidas"""
        safe_responses = [
            "jeje que divertido eres mi rey",
            "ay papasito me haces reir mucho",
            "uy corazon que manera de alegrar el dia",
            "quiubo amor eres tan especial para mi",
            "jeje mi vida que bueno hablar contigo",
            "ay que lindo momento pasamos juntos",
            "uy me encanta esta conversacion contigo",
            "quiubo corazon sigamos disfrutando esto",
            "jeje eres unico mi rey de verdad",
            "ay papasito que bonito es conocerte"
        ]
        return random.choice(safe_responses)

# --- BASE DE DATOS MEJORADA ---
DATABASE_URL = os.getenv("DATABASE_URL")

def get_db_connection():
    return psycopg2.connect(DATABASE_URL)

def init_db():
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_conversations (
                    client_id VARCHAR(255) PRIMARY KEY,
                    history JSONB,
                    message_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                );
            """)
            conn.commit()
        conn.close()
        logging.info("Base de datos inicializada")
    except Exception as e:
        logging.error(f"Error en BD: {e}")

# --- CLIENTES ACTIVOS ---
ACTIVE_CLIENTS = set()
clients_lock = threading.Lock()

def load_clients():
    try:
        with open("users.txt", 'r', encoding='utf-8') as f:
            clients = {line.strip() for line in f if line.strip() and not line.strip().startswith(('//', '#'))}
        
        with clients_lock:
            global ACTIVE_CLIENTS
            ACTIVE_CLIENTS = clients
            logging.info(f"Clientes activos cargados: {len(ACTIVE_CLIENTS)}")
    except FileNotFoundError:
        logging.warning("Archivo users.txt no encontrado")

def update_clients_periodically():
    while True:
        load_clients()
        time.sleep(60)

# --- MANEJO DE HISTORIAL POR CLIENT_ID ---
def get_user_history(client_id):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT history, message_count FROM user_conversations WHERE client_id = %s;", (client_id,))
            r = cur.fetchone()
            if r:
                history, message_count = r
                return {
                    "history": history if history else [],
                    "message_count": message_count,
                    "is_new_user": message_count == 0
                }
            return {"history": [], "message_count": 0, "is_new_user": True}
    finally:
        conn.close()

def save_user_history(client_id, history):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Mantener solo los últimos 4 mensajes (2 intercambios)
            if len(history) > 8:  # 4 mensajes de usuario + 4 del bot
                history = history[-8:]
            
            message_count = len(history) // 2  # Contar pares de mensajes
            
            cur.execute("""
                INSERT INTO user_conversations (client_id, history, message_count)
                VALUES (%s, %s, %s)
                ON CONFLICT (client_id)
                DO UPDATE SET 
                    history = EXCLUDED.history, 
                    message_count = EXCLUDED.message_count,
                    updated_at = NOW();
            """, (client_id, json.dumps(history), message_count))
            conn.commit()
    finally:
        conn.close()

# --- FLASK APP ---
app = Flask(__name__)
hybrid_chat = HybridChat()

@app.route("/")
def health_check():
    return {"status": "active", "service": "ChatBot Mejorado", "timestamp": datetime.utcnow().isoformat()}

@app.route("/chat", methods=["POST"])
def handle_chat():
    try:
        data = request.get_json()
        if not data:
            return "JSON inválido", 400
        
        user_message = data.get("message", "").strip()
        client_id = data.get("client_id")

        if not client_id:
            return "Falta client_id", 401
        
        with clients_lock:
            if client_id not in ACTIVE_CLIENTS:
                return "Suscripción inactiva", 403
        
        if not user_message:
            return "Falta mensaje", 400
        
        # Limpiar mensaje
        user_message = re.sub(r'[\r\n]+', ' ', user_message).strip()
        
        # Obtener historial del CLIENT_ID específico
        user_data = get_user_history(client_id)
        conversation_history = user_data.get("history", [])
        is_new_user = user_data.get("is_new_user", True)
        
        # Obtener respuesta
        start_time = time.time()
        response = hybrid_chat.get_response(user_message, conversation_history, is_new_user)
        response_time = time.time() - start_time
        
        # Actualizar historial (solo últimos 4 mensajes)
        conversation_history.append({"role": "USER", "message": user_message})
        conversation_history.append({"role": "CHATBOT", "message": response})
        save_user_history(client_id, conversation_history)
        
        # Log informativo
        word_count = len(response.split())
        user_type = "NUEVO" if is_new_user else "EXISTENTE"
        logging.info(f"Usuario {user_type} [{client_id[:8]}] - Respuesta: {word_count} palabras - Tiempo: {response_time:.2f}s")
        
        return response

    except Exception as e:
        logging.error(f"Error en /chat: {e}")
        return "uy se me fue el hilo repitemelo"

# --- INICIO RÁPIDO ---
if __name__ == "__main__":
    # Inicialización rápida
    init_db()
    load_clients()
    
    # Hilo para actualizar clientes
    update_thread = threading.Thread(target=update_clients_periodically, daemon=True)
    update_thread.start()
    
    # Servir
    port = int(os.environ.get("PORT", 8080))
    logging.info(f"🤖 Servicio ChatBot Mejorado iniciado en puerto {port}")
    serve(app, host="0.0.0.0", port=port, threads=10)
