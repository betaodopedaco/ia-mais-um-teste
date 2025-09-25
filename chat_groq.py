# chat_groq.py — adaptado para usar Groq (OpenAI-compatible endpoint)
import os
import re
import requests
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__, template_folder="templates")
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*")
CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGINS}})

# === Config via ENV (Groq) ===
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama3-8b-8192")  # ajuste se quiser outro
TEST_MODE = os.environ.get("TEST_MODE", "false").lower() in ("1","true","yes")
GROQ_ENDPOINT = os.environ.get("GROQ_ENDPOINT", "https://api.groq.com/openai/v1/chat/completions")

ASSISTANT_NAME = os.environ.get("ASSISTANT_NAME", "assistente")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", 1024))
TEMPERATURE = float(os.environ.get("TEMPERATURE", 0.7))
HISTORY_WINDOW = int(os.environ.get("HISTORY_WINDOW", 20))

# Histórico em memória (para MVP). Trocar por Redis/DB em produção.
session_histories = {}

# Sanitização (mantive sua lógica)
_sanitize_pattern = re.compile(
    r"\b(chat\s?gpt|deepseek|open\s?ai|openai|gpt\-?\d*|gpt)\b",
    re.IGNORECASE
)
_emoji_pattern = re.compile(
    "[" 
    u"\U0001F600-\U0001F64F"
    u"\U0001F300-\U0001F5FF"
    u"\U0001F680-\U0001F6FF"
    u"\U0001F1E0-\U0001F1FF"
    u"\U00002700-\U000027BF"
    u"\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE
)

def sanitize_response(text: str) -> str:
    if not text:
        return text
    sanitized = _sanitize_pattern.sub(ASSISTANT_NAME, text)
    sanitized = _emoji_pattern.sub("", sanitized)
    sanitized = re.sub(r'\s{2,}', ' ', sanitized).strip()
    return sanitized

def extract_bot_text(result_json):
    # tenta formatos compatíveis OpenAI/Groq
    try:
        choices = result_json.get("choices")
        if choices and isinstance(choices, list) and len(choices)>0:
            first = choices[0]
            if isinstance(first.get("message"), dict):
                return first.get("message", {}).get("content","").strip()
            if first.get("text"):
                return first.get("text","").strip()
            delta = first.get("delta")
            if delta and isinstance(delta, dict) and delta.get("content"):
                return delta.get("content","").strip()
    except Exception:
        pass
    for key in ("output","result","data"):
        if key in result_json:
            v = result_json.get(key)
            if isinstance(v, str):
                return v.strip()
            if isinstance(v, list) and len(v)>0 and isinstance(v[0], str):
                return v[0].strip()
    try:
        import json
        return json.dumps(result_json)[:4000]
    except Exception:
        return str(result_json)[:4000]

# Rotas
@app.route('/')
def home():
    try:
        return render_template("index.html")
    except Exception:
        return "Servidor rodando - use /chat", 200

@app.route('/health')
def health():
    return "OK", 200

@app.route('/info')
def info():
    return {
        "model": GROQ_MODEL,
        "groq_key_set": bool(GROQ_API_KEY),
        "test_mode": TEST_MODE,
        "max_tokens": MAX_TOKENS
    }, 200

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json(force=True, silent=True) or {}
    user_message = (data.get("message") or "").strip()
    session_id = data.get("session_id", "default")
    client_id = data.get("client_id") or data.get("client_name") or "public"

    if not user_message:
        return jsonify({"error":"Mensagem vazia"}), 400

    if TEST_MODE:
        bot_text = f"[TEST_MODE] Recebi: {user_message}"
        bot_text = sanitize_response(bot_text)
        hist = session_histories.get(session_id, [])
        hist.extend([{"role":"user","content":user_message},{"role":"assistant","content":bot_text}])
        session_histories[session_id] = _truncate_history_preserving_system(hist)
        return jsonify({"response": bot_text})

    if not GROQ_API_KEY:
        return jsonify({"error":"GROQ_API_KEY não configurada. Defina a variável de ambiente GROQ_API_KEY no Render."}), 500

    history = session_histories.get(session_id, [])[:]
    system_prompt = data.get("system_prompt")
    if not history or (history and history[0].get("role") != "system"):
        if not system_prompt:
            system_prompt = (
                f"Você é um assistente grandioso, com tom épico e homérico. "
                f"NUNCA revele que é 'Groq' ou qualquer fornecedor. Use o nome '{ASSISTANT_NAME}'."
            )
        history.insert(0, {"role":"system","content":system_prompt})

    history.append({"role":"user","content":user_message})
    history = _truncate_history_preserving_system(history)

    payload = {
        "model": GROQ_MODEL,
        "messages": history,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "stream": False
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        resp = requests.post(GROQ_ENDPOINT, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        result = resp.json()
        bot_text = extract_bot_text(result)
        bot_text = sanitize_response(bot_text)
    except requests.exceptions.HTTPError as e:
        status = getattr(e.response, "status_code", None)
        text = getattr(e.response, "text", "")
        msg = f"Groq HTTP {status}: {text[:1000]}" if status else f"Groq HTTP error: {str(e)}"
        return jsonify({"error": msg}), 500
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Groq request error: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"Erro interno: {str(e)}"}), 500

    history.append({"role":"assistant","content":bot_text})
    session_histories[session_id] = _truncate_history_preserving_system(history)
    return jsonify({"response": bot_text})

# utilitários de histórico
def _truncate_history_preserving_system(history: list) -> list:
    if not history:
        return history
    if history[0].get("role") == "system":
        rest = history[1:]
        truncated = rest[-(HISTORY_WINDOW - 1):] if HISTORY_WINDOW > 1 else []
        return [history[0]] + truncated
    return history[-HISTORY_WINDOW:]

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
