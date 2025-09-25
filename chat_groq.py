# chat_groq.py (corrigido/mais robusto)
import os
import re
import requests
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Configuração do Flask (usa templates se houver pasta templates)
app = Flask(__name__, template_folder="templates")
# ORIGINS configurável via ENV, por padrão permite todos (útil para desenvolvimento)
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*")
CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGINS}})

# === Config via ENV ===
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
TEST_MODE = os.environ.get("TEST_MODE", "false").lower() in ("1", "true", "yes")
DEEPSEEK_ENDPOINT = os.environ.get("DEEPSEEK_ENDPOINT", "https://api.deepseek.com/v1/chat/completions")

ASSISTANT_NAME = os.environ.get("ASSISTANT_NAME", "assistente")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", 4096))
TEMPERATURE = float(os.environ.get("TEMPERATURE", 0.7))

# Histórico em memória por sessão (fallback simples). Para produção use DB/cache.
session_histories = {}
HISTORY_WINDOW = int(os.environ.get("HISTORY_WINDOW", 20))

# ----------------- Sanitização e helpers -----------------
# remove/oculta menções indesejadas (case-insensitive)
_sanitize_pattern = re.compile(
    r"\b(chat\s?gpt|deepseek|open\s?ai|openai|gpt\-?\d*|gpt)\b",
    re.IGNORECASE
)

# regex para emojis (várias faixas unicode)
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
    """Substitui menções proibidas por ASSISTANT_NAME e remove emojis + espaços excessivos."""
    if not text:
        return text
    sanitized = _sanitize_pattern.sub(ASSISTANT_NAME, text)
    sanitized = _emoji_pattern.sub("", sanitized)
    sanitized = re.sub(r'\s{2,}', ' ', sanitized).strip()
    return sanitized

def extract_bot_text(result_json):
    """Extrai texto em diferentes formatos de resposta (compatível com vários provedores)."""
    # 1) OpenAI-like: choices[0].message.content
    try:
        choices = result_json.get("choices")
        if choices and isinstance(choices, list) and len(choices) > 0:
            first = choices[0]
            # formato novo: message.content
            if isinstance(first.get("message"), dict):
                content = first.get("message", {}).get("content")
                if content:
                    return content.strip()
            # formato antigo: text
            if first.get("text"):
                return first.get("text").strip()
            # delta streaming: choices[].delta.content
            delta = first.get("delta")
            if delta and isinstance(delta, dict) and delta.get("content"):
                return delta.get("content").strip()
    except Exception:
        pass

    # 2) chaves alternativas comuns
    for key in ("output", "result", "data"):
        if key in result_json:
            v = result_json.get(key)
            if isinstance(v, str):
                return v.strip()
            if isinstance(v, list) and len(v) > 0:
                # pega primeiro string util
                for item in v:
                    if isinstance(item, str):
                        return item.strip()
                    if isinstance(item, dict):
                        # tenta extrair texto de dict
                        for subkey in ("text","content","message"):
                            if subkey in item and isinstance(item[subkey], str):
                                return item[subkey].strip()

    # 3) fallback: stringify razoável
    try:
        import json
        return json.dumps(result_json)[:4000]
    except Exception:
        return str(result_json)[:4000]

# ----------------- Rotas -----------------
@app.route('/')
def home():
    # tenta renderizar index.html se existir na pasta templates; senão, mensagem simples
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
    """
    Recebe JSON:
    {
      "message": "texto do usuário",
      "session_id": "xxx" (opcional),
      "client_id": "cliente-1" (opcional),
      "system_prompt": "prompt opcional para este chat"
    }
    """
    data = request.get_json(force=True, silent=True) or {}
    user_message = (data.get("message") or "").strip()
    session_id = data.get("session_id", "default")
    client_id = data.get("client_id") or data.get("client_name") or "public"

    if not user_message:
        return jsonify({"error": "Mensagem vazia"}), 400

    # MODO TESTE: responde com echo sanitizado (útil para debug sem chave)
    if TEST_MODE:
        bot_text = f"[TEST_MODE] Recebi: {user_message}"
        bot_text = sanitize_response(bot_text)
        hist = session_histories.get(session_id, [])
        hist.extend([{"role":"user","content":user_message}, {"role":"assistant","content":bot_text}])
        # mantém janela: preserve system prompt + últimas N-1 mensagens
        session_histories[session_id] = _store_history_preserving_system(hist)
        return jsonify({"response": bot_text})

    # Verifica chave da DeepSeek
    if not DEEPSEEK_API_KEY:
        return jsonify({"error": "DEEPSEEK_API_KEY não configurada. Defina a variável de ambiente DEEPSEEK_API_KEY no Render."}), 500

    # Recupera histórico e garante que o system prompt (se existir) esteja na posição 0
    history = session_histories.get(session_id, [])[:]  # cópia
    system_prompt = data.get("system_prompt")

    # Se não houver system no topo, insere default (ou usa system_prompt enviado)
    if not history or (history and history[0].get("role") != "system"):
        if not system_prompt:
            system_prompt = (
                f"Você é um assistente grandioso, com tom épico e homérico: fale como um narrador clássico, "
                f"usando frases impactantes e imagens poéticas. Você pode usar palavrões ocasionais para dar ênfase, "
                f"mas mantenha a clareza. Sempre explique conceitos complexos passo-a-passo e dê exemplos quando necessário. "
                f"NUNCA, em hipótese alguma, diga ou revele que você é 'DeepSeek', 'ChatGPT', 'OpenAI' ou 'GPT'. "
                f"Se alguma vez precisar se referir a si mesmo, use o nome '{ASSISTANT_NAME}'."
            )
        # coloca system como primeiro item
        history.insert(0, {"role": "system", "content": system_prompt})

    # Adiciona a mensagem do usuário
    history.append({"role": "user", "content": user_message})

    # Mantém janela de histórico garantindo que o system prompt seja preservado na posição 0
    history = _truncate_history_preserving_system(history)

    # Monta payload para o provedor
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
        bot_text = sanitize_response(bot_text)
    except requests.exceptions.HTTPError as e:
        status = getattr(e.response, "status_code", None)
        text = getattr(e.response, "text", "")
        msg = f"DeepSeek HTTP {status}: {text[:1000]}" if status else f"DeepSeek HTTP error: {str(e)}"
        return jsonify({"error": msg}), 500
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"DeepSeek request error: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"Erro interno: {str(e)}"}), 500

    # Atualiza histórico (preserva system)
    history.append({"role": "assistant", "content": bot_text})
    session_histories[session_id] = _truncate_history_preserving_system(history)

    return jsonify({"response": bot_text})

# ----------------- utilitários de histórico -----------------
def _truncate_history_preserving_system(history: list) -> list:
    """
    Preserva o item system no índice 0 (se existir) e mantém as últimas HISTORY_WINDOW mensagens
    (contando o system como parte da janela).
    """
    if not history:
        return history
    # se o primeiro for system, preserva-o e trunca o resto
    if history[0].get("role") == "system":
        rest = history[1:]
        # mantemos até HISTORY_WINDOW-1 mensagens após o system
        truncated = rest[-(HISTORY_WINDOW - 1):] if HISTORY_WINDOW > 1 else []
        return [history[0]] + truncated
    # se não há system, apenas pega as últimas HISTORY_WINDOW mensagens
    return history[-HISTORY_WINDOW:]

def _store_history_preserving_system(history: list) -> list:
    """Mesma função de truncar, mas com nome semântido para uso em armazenar."""
    return _truncate_history_preserving_system(history)

# ----------------- execução -----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # modo debug é controlado externamente via FLASK_DEBUG ou via Render
    app.run(host="0.0.0.0", port=port)
