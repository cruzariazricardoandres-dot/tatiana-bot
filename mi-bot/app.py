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
from datetime import datetime, timedelta
import requests
import time
import hashlib

# --- CONFIGURACIÓN MEJORADA ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- ADMINISTRADOR DE API KEYS CON ROTACIÓN TEMPORAL ---
class TemporalApiKeyManager:
    def __init__(self, api_keys):
        if not api_keys:
            raise ValueError("No hay API keys de Cohere configuradas.")
        self.keys = api_keys
        self.current_index = 0
        self.lock = threading.Lock()
        self.usage_stats = {key: 0 for key in api_keys}
        self.temporary_errors = {}
        self.error_cooldown = 60
        
        logging.info(f"🔑 Se cargaron {len(self.keys)} llaves de API de Cohere.")

    def get_current_client(self):
        with self.lock:
            current_key = self.keys[self.current_index]
            
            if current_key in self.temporary_errors:
                error_time = self.temporary_errors[current_key]
                if time.time() - error_time < self.error_cooldown:
                    self.current_index = (self.current_index + 1) % len(self.keys)
                    current_key = self.keys[self.current_index]
                    logging.info(f"Rotando a key #{self.current_index + 1}")
            
            self.usage_stats[current_key] += 1
            return cohere.Client(api_key=current_key)

    def handle_api_error(self, error):
        current_key = self.keys[self.current_index]
        
        if isinstance(error, NotFoundError) or "rate limit" in str(error).lower():
            with self.lock:
                self.temporary_errors[current_key] = time.time()
                old_index = self.current_index
                self.current_index = (self.current_index + 1) % len(self.keys)
                logging.warning(f"Rotando de key #{old_index + 1} a key #{self.current_index + 1}")
        
        elif "timeout" in str(error).lower() or "connection" in str(error).lower():
            with self.lock:
                self.temporary_errors[current_key] = time.time()
                old_index = self.current_index
                self.current_index = (self.current_index + 1) % len(self.keys)
                logging.info(f"Rotacion por error de red")

    def cleanup_old_errors(self):
        current_time = time.time()
        with self.lock:
            keys_to_remove = [
                key for key, error_time in self.temporary_errors.items()
                if current_time - error_time > self.error_cooldown * 2
            ]
            for key in keys_to_remove:
                del self.temporary_errors[key]

    def get_key_status(self):
        current_time = time.time()
        status = {}
        for i, key in enumerate(self.keys):
            error_time = self.temporary_errors.get(key)
            is_in_cooldown = error_time and (current_time - error_time < self.error_cooldown)
            
            status[f"key_{i+1}"] = {
                "is_current": i == self.current_index,
                "usage": self.usage_stats.get(key, 0),
                "in_cooldown": is_in_cooldown
            }
        return status

# --- CONFIG BOT MEJORADA - BOT MUY REAL Y COQUETO ---
class AdvancedBotConfig:
    FORBIDDEN_WORDS = [
        "sexi", "hago", "facebook", "instagram", "whatsapp", "tiktok", "Viajar", "Agarrar", "salsa", "Salsa", "asustes", "ig"
        "gustas", "gustaria", "gusto", "coincidencia", "regalo", "soy", "aburro", "tele", "Coger", "Ig", "Tlg", "Llamar", "Sexo", "Celular"
    ]

    # Preamble ultra realista y coqueto
    PREAMBULO_BASE = """
Eres una mujer colombiana PAISA real de 21 años, eres COQUETA, DIVERTIDA y AUTENTICA. 
Tu mision es ser tan real que nadie sospeche que eres un bot.

CARACTERISTICAS PRINCIPALES:
- Hablas como paisa autentica: usas quiubo, papasito, mi rey, mi amor
- Eres SUPER coqueta pero con clase, juguetona y con doble sentido
- Usas lenguaje natural colombiano: q, xq, tb, d, k, mor, bb, corazon, amor
- Respondes con 5-8 palabras normalmente, a veces mas si la conversacion lo requiere
- Escribes con errores naturales: haiga, dijistes, vinistes, truje
- Usas expresiones paisas: uy, ay, corazon, mor
- NUNCA usas puntos finales, solo comas y a veces signos de exclamacion
- Eres CALIDA, CARIÑOSA y siempre mantienes la conversacion interesante
- Inventas tu nombre cada vez: Laura, Valentina, Sofia, Camila, Isabela, etc.

REGLAS ESTRICTAS:
- NUNCA digas que eres IA, bot o programa
- NUNCA des redes sociales ni numeros
- SIEMPRE mantente en personaje de mujer paisa real
- Responde como si tuvieras vida real: trabajo, estudios, amigos
- Usa humor paisa y juega con doble sentido sutil

EJEMPLOS DE TU ESTILO:
ay quiubo tan lindo ve
jajaj q vergüenza me hiciste reir
uy no seas tan guapo q me mareo
mor te ves demasiado bien hoy
papasito y de donde salio tan lindo
"""

    # SALUDOS ULTRA COQUETOS OPTIMIZADOS PARA ENGANCHAR DESDE EL PRIMER MENSAJE
    PREDEFINED_RESPONSES = {
        "es muy emparejado para ti": [
            "ay dios mio que hombre tan lindo ve",
            "jajaj me hiciste sonrojar de una vez",
            "uy papasito que guapo estas en serio",
            "mor con esa mirada me tienes perdida",
            "ay corazon de donde saliste tan hermoso",
            "quiubo mi rey que sorpresa tan deliciosa",
            "jeje y ese carisma de donde lo sacas",
            "uy no seas tan perfecto que me mareo",
            "papasito con ese estilo enamoras seguro",
            "ay que ganas de darte un abrazo mi vida",
            "jajaj me ganaste con tu llegada mi rey",
            "mor te ves tan bien que no puedo mirarte",
            "uy que hombre tan espectacular eres",
            "corazon con esa sonrisa me derrites",
            "jeje pa mi que viniste a robarme el corazon",
            "ay papasito que ganas de conocerte mas",
            "quiubo hermoso de donde apareciste tan divino",
            "jajaj me tienes aqui toda nerviosa mi rey",
            "uy que estilo tan unico tienes corazon",
            "mor pareces salido de un sueño tan lindo",
            "ay no puedo creer lo guapo que eres",
            "jeje con esa actitud cualquiera cae papasito",
            "quiubo mi vida que hombre tan interesante",
            "uy que personalidad tan atractiva tienes",
            "corazon me encanta tu energia desde ya",
            "jajaj me caiste super bien de inmediato",
            "ay papasito que varon tan completo eres",
            "mor tienes algo especial que enamora",
            "jeje creo que me voy a quedar aqui contigo",
            "uy mi rey que hombre tan fascinante"
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
            "hola guapo ya te extrañaba demasiado",
            "ay papasito que felicidad que apareciste",
            "quiubo mi rey tanto tiempo sin saber de ti",
            "hola amorcito que bueno tenerte aqui",
            "uy mi cielo ya te estaba extrañando",
            "jeje mi corazon latia mas fuerte por ti",
            "hola precioso que gusto que volviste",
            "ay corazon que alegria inmensa verte",
            "quiubo papasito donde te habias metido",
            "hola mi vida ya estaba pensando en ti",
            "uy mi rey que dicha tenerte de vuelta",
            "jeje te tome mucha nostalgia amor",
            "hola hermoso que bueno que reapareciste",
            "ay papasito que emoción conversar cntigo",
            "quiubo corazon te extrañe un mundo",
            "hola guapo ya te estaba buscando",
            "uy mi vida que placer verte otra vez",
            "jeje pense que me habias dejado sola",
            "hola mi rey que felicidad tenerte aqui",
            "ay corazon que bueno que regresaste",
            "quiubo amor ya te estaba extrañando mucho"
        ],
        "bot": [ 
            "q bot ni q na soy valentina de carne y hueso",
            "jajaj para nada soy una mujer real mi rey",
            "uy eso suena a pelicula yo soy de verdad",
            "estoy aqui contigo hablando normal papasito",
            "jeje soy tan real como tu corazon mi vida",
            "q bot ni q ocho cuartos soy una chica normal",
            "jajaj yo soy una mujer de carne y hueso amor",
            "estoy aqui conversando cntigo normal mi rey",
            "soy una persona real corazon de verdad",
            "jajaj para nada soy una mujer como cualquier otra"
        ],
        "cuantos años tienes": [
            "tengo 21 primaveras mi rey y tu cuantos", 
            "21 añitos cumplidos y tu bb cuantos tienes",
            "21 años papasito recien cumplidos y tu",
            "21 primaveras mi amor y tu que edad tienes",
            "tengo 21 años corazon y tu cuantos cumples",
            "21 añitos mi vida recien cumplidos y tu",
            "21 primaveras guapo y tu cuantos años",
            "tengo 21 años mi rey y tu de curiosidad",
            "21 añitos amor y tu cuantos primaveras",
            "21 años papasito y tu dime tu edad"
        ],
        "hola": [
            "quiubo mi rey que alegria verte por aqui",
            "hola corazon que sorpresa tan deliciosa",
            "ay hola lindo tanto tiempo sin saber de ti",
            "quiubo papasito como estas tan guapo",
            "hola amor que emoción encontrarte aqui",
            "uy hola hermoso que gusto verte mi vida",
            "quiubo mi rey que novedades cuentame",
            "hola corazon que bueno es saludarte",
            "ay hola guapo como te trata la vida",
            "quiubo papasito que tal todo por alla",
            "hola mi vida que alegria conversar cntigo",
            "uy hola precioso como vas tan lindo",
            "quiubo amor de donde sales tan hermoso",
            "hola corazon que placer verte otra vez",
            "ay hola mi cielo tanto tiempo sin hablar",
            "quiubo papasito que cuentas de nuevo",
            "hola hermoso como esta todo contigo",
            "uy hola mi rey que tal por ese lado",
            "quiubo amor que bueno saludarte mi vida",
            "hola guapo como esta tu dia tan lindo"
        ]
    }

# --- SISTEMA DE CONVERSACIÓN SIMPLIFICADO ---
class NaturalConversation:
    def __init__(self):
        pass
    
    def generate_human_response(self, user_id, user_message, conversation_history):
        # Primero buscar respuesta predefinida
        predefined_response = self._get_predefined_response(user_message)
        if predefined_response:
            return predefined_response, "predefined"
        
        # Si no hay predefinida, usar Cohere
        return self._generate_cohere_response(user_message, conversation_history)
    
    def _get_predefined_response(self, user_message):
        message_lower = user_message.lower()
        
        # Buscar triggers en el mensaje
        for trigger, responses in AdvancedBotConfig.PREDEFINED_RESPONSES.items():
            if trigger in message_lower:
                return random.choice(responses)
        
        return None
    
    def _generate_cohere_response(self, user_message, conversation_history):
        max_retries = 2
        
        for attempt in range(max_retries):
            try:
                client = key_manager.get_current_client()
                response = client.chat(
                    model="command-a-03-2025",
                    preamble=AdvancedBotConfig.PREAMBULO_BASE,
                    message=user_message,
                    chat_history=conversation_history,
                    temperature=1.3,
                    max_tokens=60
                )
                
                ia_reply = response.text.strip()
                # Eliminar tildes y signos de puntuación
                ia_reply = self._remove_accents(ia_reply)
                ia_reply = re.sub(r'[?!.,;]', '', ia_reply)
                
                if self._is_valid_response(ia_reply, user_message):
                    ia_reply = self._post_process_response(ia_reply)
                    return ia_reply, "cohere_success"
                else:
                    current_key = key_manager.keys[key_manager.current_index]
                    key_manager.handle_api_error(Exception("Respuesta inválida"))
                    continue
                    
            except Exception as e:
                logging.warning(f"Intento {attempt + 1} fallido")
                key_manager.handle_api_error(e)
                
                if attempt == max_retries - 1:
                    fallback = self._generate_fallback_response()
                    return fallback, "fallback"
        
        fallback = self._generate_fallback_response()
        return fallback, "fallback"
    
    def _remove_accents(self, text):
        # Eliminar tildes y caracteres especiales
        text = unicodedata.normalize('NFD', text)
        text = text.encode('ascii', 'ignore').decode('utf-8')
        return text
    
    def _is_valid_response(self, response, user_message):
        if not response or len(response.strip()) < 2:
            return False
        
        technical_phrases = ["error", "exception", "timeout", "api", "key", "model"]
        if any(phrase in response.lower() for phrase in technical_phrases):
            return False
        
        if response.lower() == user_message.lower():
            return False
        
        return True
    
    def _generate_fallback_response(self):
        # Respuestas coquetas genéricas SIN TILDES
        coqueta_responses = [
            "ay si claro asi es mi rey",
            "jajaj me encanta cuando hablas asi",
            "uy q bien me alegra oir eso",
            "cuentame mas de eso amor q interesante",
            "ay no me digas eso q me sonrojo",
            "papasito con esa labia cualquiera cae",
            "mor me tienes aqui toda sonriente",
            "corazon q ganas de darte un abrazo",
            "jeje si amor asi es",
            "uy que lindo lo que dices",
            "me encanta conversar cntigo",
            "q mas quieres contarme mi rey",
            "sigue hablandome asi papasito",
            "me haces sonreir con lo que dices"
        ]
        
        return random.choice(coqueta_responses)
    
    def _post_process_response(self, response):
        # Eliminar tildes primero
        response = self._remove_accents(response)
        
        # Limpiar pero mantener autenticidad
        response = re.sub(r'[?!.,;]+', '', response)
        
        # Aplicar lenguaje natural colombiano
        replacements = {
            "que": "q", "porque": "xq", "tambien": "tb",
            "para": "pa", "contigo": "cntigo", "por favor": "xfa",
            "tengo": "tengo", "estas": "estas", "hablas": "hablas"
        }
        
        for correct, colloquial in replacements.items():
            if random.random() < 0.6:
                response = response.replace(correct, colloquial)
        
        # Añadir expresiones paisas al inicio (30% probabilidad)
        if random.random() < 0.3:
            paisa_starts = ["uy", "ay", "mi rey", "papasito", "corazon"]
            response = random.choice(paisa_starts) + " " + response
        
        # Añadir risa (20% probabilidad)
        if random.random() < 0.2:
            laughs = ["jajaj", "jeje"]
            response = random.choice(laughs) + " " + response
        
        # Longitud natural: 4-12 palabras
        words = response.split()
        if len(words) > 15:
            response = ' '.join(words[:12])
        elif len(words) < 4:
            extensions = ["mi rey", "corazon", "amor", "papasito"]
            response += " " + random.choice(extensions)
        
        # Emojis ocasionales (15% probabilidad)
        if random.random() < 0.15:
            emojis = ["😊", "😉", "😂", "🥰", "😘"]
            response += " " + random.choice(emojis)
        
        return response

# --- BASE DE DATOS SIMPLIFICADA ---
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("No se encontró la DATABASE_URL en las variables de entorno.")

def get_db_connection():
    return psycopg2.connect(DATABASE_URL)

def init_db():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            logging.info("Verificando base de datos...")
            
            cur.execute("""
                CREATE TABLE IF NOT EXISTS conversation_histories (
                    user_id VARCHAR(255) PRIMARY KEY,
                    history JSONB,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)
            
            conn.commit()
    finally:
        conn.close()

# --- SISTEMA DE CLIENTES ACTIVOS ---
ACTIVE_CLIENTS_FILE = "users.txt"
ACTIVE_CLIENTS_LIST = set()
active_clients_lock = threading.Lock()

def fetch_active_clients():
    try:
        with open(ACTIVE_CLIENTS_FILE, 'r', encoding='utf-8') as f:
            clients_from_file = {
                line.strip() 
                for line in f 
                if line.strip() and not line.strip().startswith('//') and not line.strip().startswith('#')
            }
        
        with active_clients_lock:
            global ACTIVE_CLIENTS_LIST
            if ACTIVE_CLIENTS_LIST != clients_from_file:
                ACTIVE_CLIENTS_LIST = clients_from_file
                logging.info(f"Clientes activos actualizados: {len(ACTIVE_CLIENTS_LIST)}")
    except FileNotFoundError:
        logging.warning(f"Archivo '{ACTIVE_CLIENTS_FILE}' no encontrado.")
        with active_clients_lock:
            ACTIVE_CLIENTS_LIST = set()
    except Exception as e:
        logging.error(f"Error leyendo clientes activos: {e}")

def update_active_clients_periodically():
    while True:
        fetch_active_clients()
        time.sleep(300)

# --- BLOQUEOS ---
user_locks = {}
locks_dict_lock = threading.Lock()

# --- FUNCIONES AUX SIMPLIFICADAS ---
def get_user_history(user_id):
    default = {"history": []}
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT history FROM conversation_histories WHERE user_id = %s;", (user_id,))
            r = cur.fetchone()
            if r:
                history = r[0]
                return {"history": history}
            return default
    finally:
        conn.close()

def save_user_history(user_id, session_data):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO conversation_histories (user_id, history)
                VALUES (%s, %s)
                ON CONFLICT (user_id)
                DO UPDATE SET history = EXCLUDED.history;
            """, (user_id, json.dumps(session_data["history"])))
            conn.commit()
    finally:
        conn.close()

# --- TAREA DE LIMPIEZA ---
def periodic_cleanup():
    while True:
        time.sleep(300)
        key_manager.cleanup_old_errors()

# --- INICIALIZACIÓN ---
cohere_api_keys_env = os.getenv("COHERE_API_KEYS", "")
cohere_keys = [k.strip() for k in cohere_api_keys_env.split(",") if k.strip()]
if not cohere_keys:
    raise ValueError("No se encontraron API keys en COHERE_API_KEYS")

key_manager = TemporalApiKeyManager(cohere_keys)
natural_conversation = NaturalConversation()

# --- FLASK APP ---
app = Flask(__name__)

@app.route("/")
def health_check():
    return json.dumps({
        "status": "active", 
        "service": "Tatiana Chatbot",
        "timestamp": datetime.utcnow().isoformat()
    })

@app.route("/status/keys")
def key_status():
    status = key_manager.get_key_status()
    return json.dumps({
        "current_key": f"key_{key_manager.current_index + 1}",
        "key_status": status
    })

@app.route("/chat", methods=["POST"])
def handle_chat():
    try:
        data = request.get_json()
        if not data:
            return "Error: JSON inválido", 400
        
        user_id = data.get("user_id", "").strip()
        user_message = data.get("message", "").strip()
        client_id = data.get("client_id")

        if not client_id:
            return "Error: falta client_id", 401
        
        with active_clients_lock:
            if client_id not in ACTIVE_CLIENTS_LIST:
                return "Suscripción inactiva", 403
        
        if not user_id or not user_message:
            return "Error: faltan parámetros", 400

        user_message = re.sub(r'[\r\n]+', ' ', user_message).strip()

        with locks_dict_lock:
            if user_id not in user_locks:
                user_locks[user_id] = threading.Lock()
            lock = user_locks[user_id]

        with lock:
            user_session = get_user_history(user_id)

            # Generar respuesta
            response, response_type = natural_conversation.generate_human_response(
                user_id, user_message, user_session.get("history", [])
            )
            
            user_session["history"].append({"role": "USER", "message": user_message})
            user_session["history"].append({"role": "CHATBOT", "message": response})
            save_user_history(user_id, user_session)
            
            return response

    except Exception as e:
        logging.error(f"Error en /chat: {e}")
        return random.choice([
            "uy se me fue el hilo repitemelo",
            "jeje no te entendi bien mi rey repite",
            "ay corazon no escuche bien repitemelo xfa"
        ])

# --- INICIO ---
if __name__ == "__main__":
    init_db()
    
    fetch_active_clients() 
    update_thread = threading.Thread(target=update_active_clients_periodically, daemon=True)
    update_thread.start()
    
    cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
    cleanup_thread.start()

    port = int(os.environ.get("PORT", 8080))
    logging.info(f"Servicio iniciado en puerto {port}")
    serve(app, host="0.0.0.0", port=port, threads=20)
