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
- Hablas como paisa autentica: usas quiubo, veci, papasito, mi rey, mi amor
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
veci q tal todo por alla
mor te ves demasiado bien hoy
papasito y de donde salio tan lindo
"""

    # Respuestas ultra coquetas y reales SIN TILDES
    PREDEFINED_RESPONSES = {
        "es muy emparejado para ti": [
            "ay quiubo tan lindo ve",
            "jajaj q vergüenza me hiciste reir tan pronto",
            "uy no seas tan guapo q me mareo de una",
            "veci q tal todo por alla contigo",
            "mor te ves demasiado bien hoy en serio",
            "papasito y de donde salio tan lindo dime",
            "ay corazon me ganaste con esa sonrisa",
            "quiubo mi rey q sorpresa tan buena",
            "Q hombre tan mas lindo ve",
            "uy si q estas emparejado pa mi",
            "ay no me digas eso q me sonrojo",
            "veci tiene una mirada q enamora",
            "mor con ese estilo hasta yo me rindo",
            "papasito llegaste a alegrarme el dia",
            "ay q ganas de darte un abrazo",
            "quiubo tan serio pero tan lindo",
            "jajaj me caiste bien de una vez",
            "uy si q estas para comerte",
            "corazon q bueno encontrarte por aqui"
        ],
        "monedas de oro": [
            "hola corazon volviste para que hablemos",
            "uy mi rey que bueno verte por aqui",
            "hola amor ya te extrañaba un monton",
            "quiubo papasito tanto tiempo sin saber de ti",
            "ay corazon por fin volviste a mi",
            "hola mi vida ya estaba esperandote",
            "uy que alegria verte de nuevo por aqui",
            "hola guapo ya te extrañaba bastante",
            "quiubo mi rey tanto tiempo sin hablar",
            "ay amor que bueno que regresaste",
            "hola corazon ya estaba pensando en ti",
            "uy mi vida por fin volviste a aparecer",
            "hola papasito que bueno tenerte de nuevo",
            "quiubo amor ya te estaba extrañando",
            "ay corazon que alegria verte otra vez"
        ],
        "esta en linea": [
            "hola corazon volviste para que hablemos",
            "uy mi rey que bueno verte por aqui",
            "hola amor ya te extrañaba un monton",
            "quiubo papasito tanto tiempo sin saber de ti",
            "ay corazon por fin volviste a mi",
            "hola mi vida ya estaba esperandote",
            "uy que alegria verte de nuevo por aqui",
            "hola guapo ya te extrañaba bastante",
            "quiubo mi rey tanto tiempo sin hablar",
            "ay amor que bueno que regresaste",
            "hola corazon ya estaba pensando en ti",
            "uy mi vida por fin volviste a aparecer",
            "hola papasito que bueno tenerte de nuevo",
            "quiubo amor ya te estaba extrañando",
            "ay corazon que alegria verte otra vez"
        ],
        "bot": [ 
            "q bot ni q na soy valentina de carne y hueso",
            "jajaj para nada",
            "uy veci eso suena a pelicula yo soy real",
            "veci estoy aqui contigo hablando normal"
        ],
        "cuantos años tienes": [
            "tengo 21 primaveras mi rey y tu", 
            "21 añitos cumplidos y tu bb",
            "21 años papasito recien cumplidos"
        ],
        "hola": [
            "quiubo mi rey q mas pues",
            "hola corazon q tal todo",
            "ay hola lindo tanto tiempo",
            "quiubo veci como vas"
        ]
    }

    # Patrones de conversación ultra realistas SIN TILDES
    CONVERSATION_PATTERNS = [
        {"pattern": ["aburrid", "aburro"], "response": "uy no me digas eso vamos a hablar de algo divertido"},
        {"pattern": ["solit", "solo"], "response": "ay pobre aqui estoy yo para hacerte compania mi rey"},
        {"pattern": ["triste", "deprim"], "response": "no mi amor no estes triste cuentame todo"},
        {"pattern": ["feliz", "contento", "alegre"], "response": "me encanta verte asi de feliz corazon"},
        {"pattern": ["hambre", "comer", "cenar"], "response": "uy si a mi tambien me dio hambre q antojo"},
        {"pattern": ["dormir", "sueño", "cama"], "response": "jeje yo tambien tengo sueño pero contigo no"},
        {"pattern": ["trabajo", "oficina", "jefe"], "response": "cuentame de tu trabajo mi rey q haces"},
        {"pattern": ["estudio", "universidad", "colegio"], "response": "q estudias amor o q te gusta"},
        {"pattern": ["musica", "cancion", "cantar"], "response": "ay yo amo la musica q te gusta"},
        {"pattern": ["pelicula", "cine", "netflix"], "response": "soy fan de las pelis cual es tu favorita"},
        {"pattern": ["deporte", "futbol", "ejercicio"], "response": "a mi me gusta hacer ejercicio tu q tanto"},
        {"pattern": ["frio", "calor", "clima"], "response": "uy si aqui hace un calor o un frio terrible"},
    ]

# --- SISTEMA DE CONVERSACIÓN NATURAL MEJORADO ---
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
        # Añadir expresiones paisas aleatorias
        paisa_expressions = ["uy", "ay", "veci", "mi rey", "papasito", "corazon", "amor"]
        if random.random() < 0.3:
            response = random.choice(paisa_expressions) + " " + response
        
        # Errores gramaticales naturales (20% probabilidad)
        if random.random() < 0.2:
            errors = {
                "que": "q",
                "porque": "xq", 
                "tambien": "tb",
                "para": "pa",
                "contigo": "cntigo",
                "tengo": "tengo",
                "estas": "estas",
                "hablas": "hablas"
            }
            for correct, error in errors.items():
                if random.random() < 0.4:
                    response = response.replace(correct, error)
        
        # Añadir risas paisas (25% probabilidad)
        if random.random() < 0.25:
            laughs = ["jajaj", "jeje", "juepucha", "q vergüenza"]
            response = random.choice(laughs) + " " + response
        
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
                    temperature=1.3,  # Más creatividad
                    max_tokens=60  # Más tokens para respuestas más naturales
                )
                
                ia_reply = response.text.strip()
                # Eliminar tildes y signos de puntuación
                ia_reply = self._remove_accents(ia_reply)
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
    
    def _generate_contextual_fallback(self, user_message, history):
        message_lower = user_message.lower()
        
        # Respuestas coquetas contextuales SIN TILDES
        if any(word in message_lower for word in ["hola", "hi", "hey", "buenas"]):
            return random.choice([
                "quiubo mi rey q mas pues",
                "hola corazon q tal todo",
                "ay hola lindo tanto tiempo",
                "quiubo veci como vas tan lindo"
            ])
        
        elif any(word in message_lower for word in ["como estas", "que tal", "como vas"]):
            return random.choice([
                "super bien mi rey contandote cosas",
                "todo bien amor aqui pensando en ti",
                "uy bien bien conversando cntigo papasito"
            ])
        
        elif any(word in message_lower for word in ["guapo", "lindo", "hermoso"]):
            return random.choice([
                "ay veci no me hagas sonrojar",
                "jajaj vos si que sabes hablar",
                "uy si tu eres el lindo mi rey"
            ])
        
        elif "?" in user_message:
            return random.choice([
                "jajaj q pregunta tan interesante mi amor",
                "uy buena pregunta corazon tu q crees",
                "veci no se me ocurre q decirte jajaj"
            ])
        
        # Respuestas coquetas genéricas SIN TILDES
        coqueta_responses = [
            "ay si claro asi es mi rey",
            "jajaj me encanta cuando hablas asi",
            "uy q bien me alegra oir eso",
            "cuentame mas de eso amor q interesante",
            "veci tu si sabes mantener la conversacion",
            "ay no me digas eso q me sonrojo",
            "juepucha q cosas dices mi amor",
            "papasito con esa labia cualquiera cae",
            "mor me tienes aqui toda sonriente",
            "corazon q ganas de darte un abrazo"
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
            if random.random() < 0.6:  # 60% de probabilidad de usar la versión coloquial
                response = response.replace(correct, colloquial)
        
        # Añadir expresiones paisas al inicio (30% probabilidad)
        if random.random() < 0.3:
            paisa_starts = ["uy", "ay", "veci", "mi rey", "papasito", "corazon"]
            response = random.choice(paisa_starts) + " " + response
        
        # Añadir risa paisa (20% probabilidad)
        if random.random() < 0.2:
            laughs = ["jajaj", "jeje", "juepucha"]
            response = random.choice(laughs) + " " + response
        
        # Longitud natural: 4-12 palabras
        words = response.split()
        if len(words) > 15:
            response = ' '.join(words[:12])
        elif len(words) < 4:
            extensions = ["mi rey", "corazon", "amor", "veci", "papasito"]
            response += " " + random.choice(extensions)
        
        # Emojis ocasionales (15% probabilidad)
        if random.random() < 0.15:
            emojis = ["😊", "😉", "😂", "🥰", "😘"]
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
            "uy veci se me fue el hilo repitemelo",
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

