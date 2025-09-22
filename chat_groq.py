# chat_groq.py (modificado para DeepSeek + sanitização)
import os
import re
import requests
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__, template_folder="templates")
CORS(app, resources={r"/*": {"origins": "*"}})  # permite front separado

# === Config via ENV ===
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
TEST_MODE = os.environ.get("TEST_MODE", "false").lower() in ("1", "true", "yes")
DEEPSEEK_ENDPOINT = os.environ.get("DEEPSEEK_ENDPOINT", "https://api.deepseek.com/v1/chat/completions")

ASSISTANT_NAME = os.environ.get("ASSISTANT_NAME", "seu assistente")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", 4096))
TEMPERATURE = float(os.environ.get("TEMPERATURE", 0.7))

# Histórico em memória por sessão (fallback simples)
session_histories = {}
HISTORY_WINDOW = int(os.environ.get("HISTORY_WINDOW", 20))

# ----------------- Sanitização e helpers -----------------
# remove menções indesejadas (case-insensitive)
_sanitize_pattern = re.compile(
    r"\b(chat\s?gpt|deepseek|open\s?ai|openai|gpt\-?\d*|gpt)\b",
    re.IGNORECASE
)

# simples regex para emojis (várias faixas unicode)
_emoji_pattern = re.compile(
    "[" 
    u"\U0001F600-\U0001F64F"  # emoticons
    u"\U0001F300-\U0001F5FF"  # símbolos & pictogramas
    u"\U0001F680-\U0001F6FF"  # transportes & map
    u"\U0001F1E0-\U0001F1FF"  # bandeiras
    u"\U00002700-\U000027BF"
    u"\U000024C2-\U0001F251"
    "]+", 
    flags=re.UNICODE
)

def sanitize_response(text: str) -> str:
    """Substitui menções proibidas por ASSISTANT_NAME e remove emojis."""
    if not text:
        return text
    # Substitui referências a provedores/modelos
    sanitized = _sanitize_pattern.sub(ASSISTANT_NAME, text)
    # Remove emojis
    sanitized = _emoji_pattern.sub("", sanitized)
    # Opcional: remove sequências excessivas de espaços
    sanitized = re.sub(r'\s{2,}', ' ', sanitized).strip()
    return sanitized

def extract_bot_text(result_json):
    """Tenta extrair o texto retornado pela API DeepSeek (tenta suportar formatos comuns)."""
    # Tenta o formato tipo OpenAI/Groq: choices[0].message.content
    try:
        choices = result_json.get("choices")
        if choices and isinstance(choices, list) and len(choices) > 0:
            first = choices[0]
            # novo estilo: message -> content
            if isinstance(first.get("message"), dict):
                return first.get("message", {}).get("content", "").strip()
            # old style: text
            if first.get("text"):
                return first.get("text", "").strip()
    except Exception:
        pass
    # fallback: "output" ou "result"
    for key in ("output", "result", "data"):
        if key in result_json:
            v = result_json.get(key)
            if isinstance(v, str):
                return v.strip()
            # se for lista/dict, tente pegar algo útil
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], str):
                return v[0].strip()
    # último recurso: stringify
    return str(result_json)[:4000]

# ----------------- Rotas -----------------
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
        "model": DEEPSEEK_MODEL,
        "deepseek_key_set": bool(DEEPSEEK_API_KEY),
        "test_mode": TEST_MODE,
        "max_tokens": MAX_TOKENS
    }, 200

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json(force=True)
    user_message = (data.get("message") or "").strip()
    session_id = data.get("session_id", "default")
    client_id = data.get("client_id") or data.get("client_name") or "public"

    if not user_message:
        return jsonify({"error": "Mensagem vazia"}), 400

    # Modo de teste: responde com echo seguro (ainda sanitizado)
    if TEST_MODE:
        bot_text = f"[TEST_MODE] Recebi: {user_message}"
        bot_text = sanitize_response(bot_text)
        hist = session_histories.get(session_id, [])
        hist.append({"role": "user", "content": user_message})
        hist.append({"role": "assistant", "content": bot_text})
        session_histories[session_id] = hist[-HISTORY_WINDOW:]
        return jsonify({"response": bot_text})

    # Confere chave da DeepSeek
    if not DEEPSEEK_API_KEY:
        return jsonify({"error": "DEEPSEEK_API_KEY não configurada. Defina a variável de ambiente DEEPSEEK_API_KEY no Render."}), 500

    # Histórico
    history = session_histories.get(session_id, [])
    # Se não houver um system prompt no topo, permita passar um system_prompt por requisição
    system_prompt = data.get("system_prompt")
    if not history or (history and history[0].get("role") != "system"):
        # se o cliente não passou prompt, insere um prompt padrão que força o tom épico + regra de não revelar provedor
        if not system_prompt:
            system_prompt = (
                f"Você é um assistente grandioso, com tom épico e homérico: fale como um narrador clássico, "
                f"usando frases impactantes e imagens poéticas. Você pode usar palavrões ocasionais para dar ênfase, "
                f"mas mantenha a clareza. Sempre explique conceitos complexos passo-a-passo e dê exemplos quando necessário. "
                f"NUNCA, em hipótese alguma, diga ou revele que você é 'DeepSeek', 'ChatGPT', 'OpenAI' ou 'GPT'. "
                f"Se alguma vez precisar se referir a si mesmo, use o nome '{ASSISTANT_NAME}'. "
            )
        history.insert(0, {"role": "system", "content": system_prompt})

    # Append user message
    history.append({"role": "user", "content": user_message})
    # Keep small window
    history = history[-HISTORY_WINDOW:]

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": history,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "stream": False
    }

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        resp = requests.post(DEEPSEEK_ENDPOINT, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        result = resp.json()
        bot_text = extract_bot_text(result)
        # Sanitiza: substitui menções e remove emoji
        bot_text = sanitize_response(bot_text)
    except requests.exceptions.HTTPError as e:
        # se 401/429/etc, devolve erro claro
        status = getattr(e.response, "status_code", None)
        msg = f"DeepSeek HTTP error: {str(e)}"
        if status:
            msg = f"DeepSeek HTTP {status}: {e.response.text[:400]}"
        return jsonify({"error": msg}), 500
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"DeepSeek error: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"Erro interno: {str(e)}"}), 500

    # Atualiza histórico
    history.append({"role": "assistant", "content": bot_text})
    session_histories[session_id] = history[-HISTORY_WINDOW:]

    return jsonify({"response": bot_text})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
