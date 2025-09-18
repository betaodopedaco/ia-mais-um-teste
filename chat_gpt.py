import os
import requests
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__, template_folder="templates")
CORS(app)

# Configs por env vars
HF_MODEL = os.environ.get("HF_MODEL", "facebook/blenderbot_small-90M")
HF_TOKEN = os.environ.get("HF_TOKEN")  # obrigatório no Render
HF_API = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", 120))
TEST_MODE = os.environ.get("TEST_MODE", "false").lower() in ("1", "true", "yes")

# Histórico por sessão (memória — reinicia ao reiniciar app)
session_histories = {}

@app.route("/")
def home():
    try:
        return render_template("index.html")
    except Exception:
        return "Servidor rodando - acesse /health", 200

@app.route("/health")
def health():
    return "OK", 200

@app.route("/info")
def info():
    return {"model": HF_MODEL, "token_set": bool(HF_TOKEN), "test_mode": TEST_MODE}, 200

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    user_message = (data.get("message") or "").strip()
    session_id = data.get("session_id", "default")

    if not user_message:
        return jsonify({"error": "Mensagem vazia"}), 400

    # modo de teste local sem HF (útil pra debug)
    if TEST_MODE:
        bot_text = f"[TEST MODE] Eco: {user_message}"
        history = session_histories.get(session_id, "")
        session_histories[session_id] = (history + f"\nUser: {user_message}\nBot: {bot_text}")[-2000:]
        return jsonify({"response": bot_text})

    if not HF_TOKEN:
        return jsonify({"error": "HF_TOKEN não configurado. Defina a variável de ambiente HF_TOKEN no Render."}), 500

    prompt = (session_histories.get(session_id, "") + "\nUser: " + user_message + "\nBot:").strip()

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "do_sample": True
        }
    }

    try:
        resp = requests.post(HF_API, headers=headers, json=payload, timeout=30)
    except requests.exceptions.RequestException as e:
        return jsonify({"error": "Erro de conexão com HuggingFace: " + str(e)}), 500

    if resp.status_code == 401:
        return jsonify({"error": "Unauthorized: token inválido/expirado (401)"}), 401
    if resp.status_code == 404:
        return jsonify({"error": f"Modelo não encontrado (404). Verifique HF_MODEL: {HF_MODEL}"}), 404
    if resp.status_code >= 400:
        try:
            err = resp.json()
        except Exception:
            err = resp.text
        return jsonify({"error": f"HuggingFace API error: {resp.status_code} - {err}"}), 500

    try:
        result = resp.json()
        bot_text = ""
        if isinstance(result, dict) and "error" in result:
            return jsonify({"error": result.get("error")}), 500
        elif isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
            bot_text = result[0].get("generated_text") or result[0].get("text") or ""
        elif isinstance(result, dict) and "generated_text" in result:
            bot_text = result.get("generated_text")
        else:
            bot_text = str(result)
        bot_text = bot_text.strip()
    except Exception as e:
        return jsonify({"error": "Erro ao processar resposta da HF: " + str(e)}), 500

    # guarda histórico curto
    history = session_histories.get(session_id, "")
    new_history = (history + f"\nUser: {user_message}\nBot: {bot_text}") if history else f"User: {user_message}\nBot: {bot_text}"
    session_histories[session_id] = new_history[-2000:]

    return jsonify({"response": bot_text})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
