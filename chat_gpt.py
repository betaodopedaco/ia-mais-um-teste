# chat_gpt.py (versão Groq)
import os
import requests
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__, template_folder="templates")
CORS(app, resources={r"/*": {"origins": "*"}})  # permite front separado (temporário)

# Config
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
TEST_MODE = os.environ.get("TEST_MODE", "false").lower() in ("1", "true", "yes")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama3-70b-8192")  # modelo padrão Groq

if not GROQ_API_KEY:
    print("⚠️  GROQ_API_KEY não definido — defina em ENV vars no Render")

# histórico em memória por sessão
session_histories = {}

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
        "groq_model": GROQ_MODEL,
        "groq_key_set": bool(GROQ_API_KEY),
        "test_mode": TEST_MODE
    }, 200

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json(force=True)
    user_message = (data.get("message") or "").strip()
    session_id = data.get("session_id", "default")

    if not user_message:
        return jsonify({"error": "Mensagem vazia"}), 400

    # Test mode: devolve echo (útil pra validar front)
    if TEST_MODE:
        bot_text = f"[TEST_MODE] Recebi: {user_message}"
        hist = session_histories.get(session_id, [])
        hist.append({"role": "user", "content": user_message})
        hist.append({"role": "assistant", "content": bot_text})
        session_histories[session_id] = hist[-20:]
        return jsonify({"response": bot_text})

    # Verifica chave
    if not GROQ_API_KEY:
        return jsonify({"error": "GROQ_API_KEY não configurada. Defina a variável no Render."}), 500

    # monta mensagens
    history = session_histories.get(session_id, [])
    history.append({"role": "user", "content": user_message})

    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": GROQ_MODEL,
            "messages": history,
            "max_tokens": 200,
            "temperature": 0.7
        }
        resp = requests.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        bot_text = data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return jsonify({"error": f"Groq error: {str(e)}"}), 500

    # atualiza histórico (mantém últimos 20)
    history.append({"role": "assistant", "content": bot_text})
    session_histories[session_id] = history[-20:]

    return jsonify({"response": bot_text})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
