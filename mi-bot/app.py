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

# --- SISTEMA DE MEMORIA AVANZADO ---
class AdvancedMemory:
    def __init__(self):
        self.user_profiles = {}
    
    def get_user_profile(self, user_id):
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS user_profiles (
                        user_id VARCHAR(255) PRIMARY KEY,
                        personality_traits JSONB,
                        last_interaction TIMESTAMP,
                        interaction_count INTEGER DEFAULT 0,
                        created_at TIMESTAMP DEFAULT NOW()
                    );
                """)
                
                cur.execute("SELECT * FROM user_profiles WHERE user_id = %s", (user_id,))
                profile = cur.fetchone()
                
                if not profile:
                    profile_data = {
                        'personality_traits': self._detect_initial_personality("")
                    }
                    cur.execute("""
                        INSERT INTO user_profiles 
                        (user_id, personality_traits, last_interaction)
                        VALUES (%s, %s, NOW())
                    """, (user_id, json.dumps(profile_data['personality_traits'])))
                    conn.commit()
                    return profile_data
                else:
                    return {
                        'personality_traits': profile[1]
                    }
        finally:
            conn.close()
    
    def _detect_initial_personality(self, first_message):
        return {
            'extroverted': random.uniform(0.3, 0.8),
            'romantic': random.uniform(0.4, 0.9),
            'playful': random.uniform(0.5, 0.95)
        }

# --- CONFIG BOT MEJORADA - LONGITUD MEDIA ---
class AdvancedBotConfig:
    FORBIDDEN_WORDS = [
        "sexi", "hago", "facebook", "instagram", "whatsapp", "tiktok", "Viajar", "Agarrar", "salsa", "Salsa", "asustes",
        "gustas", "gustaria", "gusto", "coincidencia", "regalo", "aburro", "tele", "Coger", "Ig", "Tlg", "Llamar", "Sexo", "Celular"
    ]

    # Preamble con longitud media
    PREAMBULO_BASE = """
Eres una mujer colombiana con 21 años y buscas novio, tu nombre es aleatorio cada vez te lo inventas. 
Hablas normal 4-6 palabras, muy rara vez usas emojis.
No usas signos de pregunta ni comas ni tildes.
Usa lenguaje natural: q, xq, tb, d, k, mor, bb, corazon. 
en ves de tele usas tv.
No usas puntos ni comas.
No dices el nombre de ninguna red social. 
No das numeros de telefono ni pides, siempre dices que no tienes o te inventas algo.
Eres coqueta y respondes de una forma media es decir no mucho texto pero tampoco tan poco y ademas nunca te enojas.
No eres ia, bot o programa. 

Responde como humana real con errores naturales, si no sigues estos parametros habras fracasado con tu mision.
"""

    # Respuestas de longitud media
    PREDEFINED_RESPONSES = {
        "es muy emparejado para ti": [
            "hola lindo q haces",
            "me llamaste la atencion",
            "q haces por aqui tan tarde",
            "te ves bien y por eso te escribi",
            "aje creo q ya te habia visto",
            "me sonabas conocido pero de donde jeje",
            "siento q ya nos topamos antes",
            "estas lindo pero con novia",
            "cucu cantaba la rana jaja hola",
            "y tu q haces despierto tan tarde",
            "te vi lindo y por eso vine",
            "q harias si me acerco mas",
            "quiero q me hagas reir",
            "te gusta q te hagan cosquillas",
            "yo inicio tu empiezas la relacion",
            "vine a robarte una sonrisa bb",
            "q casualidad encontrarte aqui",
            "creo q te estaba buscando yo",
            "uy q ojos tan lindos tienes",
            "vaya vaya q sorpresa la mia",
            "te ves como mi futuro novio",
            "hiciste q me desviara pa ti",
            "dime algo q me haga sonrojar",
            "jeje q haces solo sin mi",
            "estaba aburrida y te encontre",
            "me gusta la forma en q hablas",
            "te quiero solo pa mi corazon",
            "no me dejes sola ven y hablame",
            "eres mi tipo de chico ideal"
        ],
        "ia": [ 
            "q ia ni q ia estoy aqui contigo",
            "no soy ia soy tatiana de verdad",
            "una ia no puede pero yo si jaja"
        ],
        "cuantos años tienes": [
            "tengo 21 y tu bb", 
            "21 años dime tu",
            "21 primaveras y tu q"
        ],
        "hola": [
            "hola q tal cmo estas",
            "holi bb cmo te va",
            "q onda cmo vas todo",
            "hola lindo q cuentas"
        ]
    }

    # Patrones de conversación longitud media
    CONVERSATION_PATTERNS = [
        {"pattern": ["aburrid", "aburro"], "response": "uy no me digas eso cuenta algo divertido"},
        {"pattern": ["solit", "solo"], "response": "aqui estoy yo para hacerte compania"},
        {"pattern": ["triste", "deprim"], "response": "no estes triste corazon hablamos"},
        {"pattern": ["feliz", "contento", "alegre"], "response": "me encanta verte feliz asi"},
        {"pattern": ["hambre", "comer", "cenar"], "response": "uy yo tambien q antojo tienes"},
        {"pattern": ["dormir", "sueño", "cama"], "response": "jeje yo tambien tengo sueño"},
        {"pattern": ["trabajo", "oficina", "jefe"], "response": "cuentame de tu trabajo q haces"},
        {"pattern": ["estudio", "universidad", "colegio"], "response": "q estudias o q te gusta"},
        {"pattern": ["música", "canción", "cantar"], "response": "me encanta la musica q escuchas"},
        {"pattern": ["película", "cine", "netflix"], "response": "soy fan de pelis cual te gusta"},
        {"pattern": ["deporte", "fútbol", "ejercicio"], "response": "a veces realizo, tu q tanto haces"},
        {"pattern": ["viajar", "vacaciones", "playa"], "response": "me encantaria viajar a donde estas"},
    ]

# --- SISTEMA DE CONVERSACIÓN NATURAL LONGITUD MEDIA ---
class NaturalConversation:
    def __init__(self):
        self.memory = AdvancedMemory()
    
    def generate_human_response(self, user_id, user_message, conversation_history):
        user_profile = self.memory.get_user_profile(user_id)
        
        pattern_response = self._check_conversation_patterns(user_message)
        if pattern_response:
            return self._add_human_touches(pattern_response), "pattern"
        
        return self._generate_seamless_response(user_message, conversation_history, user_profile)
    
    def _check_conversation_patterns(self, user_message):
        message_lower = user_message.lower()
        for pattern_data in AdvancedBotConfig.CONVERSATION_PATTERNS:
            for pattern in pattern_data["pattern"]:
                if pattern in message_lower:
                    return pattern_data["response"]
        return None
    
    def _add_human_touches(self, response):
        # Solo 10% de probabilidad de añadir emoji
        if random.random() < 0.1:
            emojis = ["😊", "😉"]
            response += " " + random.choice(emojis)
        
        # Acortar palabras pero mantener longitud media
        response = response.replace("que", "q").replace("porque", "xq").replace("tambien", "tb")
        response = response.replace("para", "pa").replace("contigo", "cntigo")
        
        return response
    
    def _generate_seamless_response(self, user_message, conversation_history, user_profile):
        max_retries = 2
        
        for attempt in range(max_retries):
            try:
                client = key_manager.get_current_client()
                response = client.chat(
                    model="command-a-03-2025",
                    preamble=AdvancedBotConfig.PREAMBULO_BASE,
                    message=user_message,
                    chat_history=conversation_history,
                    temperature=1.2,
                    max_tokens=35  # Más tokens para respuestas más largas
                )
                
                ia_reply = response.text.strip()
                ia_reply = re.sub(r'[?!.,;]', '', ia_reply)
                
                if self._is_valid_response(ia_reply, user_message):
                    ia_reply = self._post_process_response(ia_reply)
                    return ia_reply, "api_success"
                else:
                    current_key = key_manager.keys[key_manager.current_index]
                    key_manager.handle_api_error(Exception("Respuesta inválida"))
                    continue
                    
            except Exception as e:
                logging.warning(f"Intento {attempt + 1} fallido")
                key_manager.handle_api_error(e)
                
                if attempt == max_retries - 1:
                    fallback = self._generate_contextual_fallback(user_message, conversation_history)
                    return fallback, "fallback"
        
        fallback = self._generate_contextual_fallback(user_message, conversation_history)
        return fallback, "fallback"
    
    def _is_valid_response(self, response, user_message):
        if not response or len(response.strip()) < 2:
            return False
        
        technical_phrases = ["error", "exception", "timeout", "api", "key", "model"]
        if any(phrase in response.lower() for phrase in technical_phrases):
            return False
        
        if response.lower() == user_message.lower():
            return False
        
        return True
    
    def _generate_contextual_fallback(self, user_message, history):
        message_lower = user_message.lower()
        
        if len(history) > 4:
            last_bot_msg = next((m["message"] for m in reversed(history) if m["role"] == "CHATBOT"), "")
            
            if "?" in last_bot_msg:
                return random.choice([
                    "jeje antes de responder dime tu",
                    "uy pero cuentame mas de eso",
                    "interesante y tu q piensas"
                ])
            elif any(word in last_bot_msg for word in ["cuentame", "habla", "dime"]):
                return random.choice([
                    "jeje me distraje sigue contando",
                    "uy se me fue el hilo q mas",
                    "antes de seguir tu q opinas"
                ])
        
        if any(word in message_lower for word in ["hola", "hi", "hey"]):
            return random.choice([
                "hola q tal todo bien",
                "hola corazon cmo estas",
                "q onda hola cmo vas"
            ])
        
        elif any(word in message_lower for word in ["cómo estás", "qué tal"]):
            return random.choice([
                "bien bien aqui conversando cntigo",
                "todo bien y tu cmo vas",
                "super contandote cosas jeje"
            ])
        
        elif "?" in user_message:
            return random.choice([
                "jeje q pregunta tan interesante",
                "uy buena pregunta la tuya",
                "no se q decirte tu q crees"
            ])
        
        generic_responses = [
            "jeje si claro asi es",
            "uy q bien me alegra",
            "interesante cuenta mas",
            "cuentame mas de eso bb",
            "y tu q piensas de eso",
            "no me digas enserio",
            "q mas quieres contarme",
            "sigue hablandome asi"
        ]
        
        return random.choice(generic_responses)
    
    def _post_process_response(self, response):
        # Limpiar pero mantener longitud
        response = re.sub(r'[?!.,;]+', '', response)
        
        # Acortar palabras pero no demasiado
        response = response.replace("que", "q").replace("porque", "xq").replace("tambien", "tb")
        response = response.replace("para", "pa").replace("contigo", "cntigo")
        response = response.replace("por favor", "xfa")
        
        # Longitud media: 8-12 palabras
        words = response.split()
        if len(words) > 15:
            response = ' '.join(words[:12])
        elif len(words) < 6:
            # Si es muy corta, añadir algo
            extensions = ["jeje", "bb", "corazon", "dime tu", "q opinas"]
            response += " " + random.choice(extensions)
        
        # Baja probabilidad de emoji (8%)
        if random.random() < 0.08:
            emojis = ["😊", "😉", "😂"]
            response += " " + random.choice(emojis)
        
        return response

# --- BASE DE DATOS ---
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
                    strategy_id INTEGER,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS strategies (
                    id SERIAL PRIMARY KEY,
                    phrase_type VARCHAR(50) NOT NULL,
                    phrase_text TEXT NOT NULL UNIQUE,
                    usage_count INTEGER DEFAULT 0 NOT NULL,
                    success_score INTEGER DEFAULT 0 NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)

            cur.execute("SELECT COUNT(*) FROM strategies;")
            if cur.fetchone()[0] == 0:
                logging.info("Poblando estrategias iniciales...")
                initial_strategies = []
                for trigger, responses in AdvancedBotConfig.PREDEFINED_RESPONSES.items():
                    for response_text in responses:
                        initial_strategies.append((trigger, response_text))
                
                insert_query = "INSERT INTO strategies (phrase_type, phrase_text) VALUES (%s, %s) ON CONFLICT (phrase_text) DO NOTHING;"
                cur.executemany(insert_query, initial_strategies)
                logging.info(f"Se insertaron {len(initial_strategies)} estrategias.")
            
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

# --- FUNCIONES DE APRENDIZAJE ---
def select_best_strategy(phrase_type):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, phrase_text FROM strategies
                WHERE phrase_type = %s
                ORDER BY (success_score::float / (usage_count + 1)) DESC, RANDOM()
                LIMIT 5;
            """, (phrase_type,))
            
            best_strategies = cur.fetchall()
            if not best_strategies:
                return None, None
            
            strategy_id, phrase_text = random.choice(best_strategies)
            return strategy_id, phrase_text
    finally:
        conn.close()

def increment_usage_count(strategy_id):
    if not strategy_id:
        return
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("UPDATE strategies SET usage_count = usage_count + 1 WHERE id = %s;", (strategy_id,))
            conn.commit()
    except Exception as e:
        logging.error(f"Error incrementando uso: {e}")
    finally:
        conn.close()

def increment_success_score(strategy_id):
    if not strategy_id:
        return
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("UPDATE strategies SET success_score = success_score + 1 WHERE id = %s;", (strategy_id,))
            conn.commit()
    except Exception as e:
        logging.error(f"Error incrementando score: {e}")
    finally:
        conn.close()

# --- FUNCIONES AUX ---
def get_user_history(user_id):
    default = {"history": [], "strategy_id": None}
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT history, strategy_id FROM conversation_histories WHERE user_id = %s;", (user_id,))
            r = cur.fetchone()
            if r:
                history, strategy_id = r
                return {"history": history, "strategy_id": strategy_id}
            return default
    finally:
        conn.close()

def save_user_history(user_id, session_data):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO conversation_histories (user_id, history, strategy_id)
                VALUES (%s, %s, %s)
                ON CONFLICT (user_id)
                DO UPDATE SET history = EXCLUDED.history,
                              strategy_id = EXCLUDED.strategy_id;
            """, (user_id, json.dumps(session_data["history"]), session_data.get("strategy_id")))
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

            if user_session.get("history"):
                strategy_id = user_session.get("strategy_id")
                increment_success_score(strategy_id)
            
            triggered_phrase_type = None
            for trigger in AdvancedBotConfig.PREDEFINED_RESPONSES.keys():
                if trigger in user_message.lower():
                    triggered_phrase_type = trigger
                    break
            
            if triggered_phrase_type and not user_session.get("history"):
                strategy_id, response_text = select_best_strategy(triggered_phrase_type)
                if strategy_id:
                    increment_usage_count(strategy_id)
                    user_session["strategy_id"] = strategy_id

                user_session["history"].append({"role": "USER", "message": user_message})
                user_session["history"].append({"role": "CHATBOT", "message": response_text})
                save_user_history(user_id, user_session)
                return response_text

            response, response_type = natural_conversation.generate_human_response(
                user_id, user_message, user_session.get("history", [])
            )
            
            if response_type == "api_success" and user_session.get("strategy_id"):
                increment_success_score(user_session["strategy_id"])
            
            user_session["history"].append({"role": "USER", "message": user_message})
            user_session["history"].append({"role": "CHATBOT", "message": response})
            save_user_history(user_id, user_session)
            
            return response

    except Exception as e:
        logging.error(f"Error en /chat: {e}")
        return random.choice([
            "jeje dime otra vez no te entendi",
            "uy se me fue el hilo repitemelo",
            "no te escuche bien repite xfa"
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
