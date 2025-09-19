# chat_gpt.py
import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__, template_folder="templates")
CORS(app, resources={r"/*": {"origins": "*"}})  # permite front separado (temporário)

# Config
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
TEST_MODE = os.environ.get("TEST_MODE", "false").lower() in ("1", "true", "yes")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")  # pode trocar depois

# cliente OpenAI (usa biblioteca oficial)
if OPENAI_API_KEY:
    try:
        import openai
        openai.api_key = OPENAI_API_KEY
    except Exception as e:
        print("Erro importando openai:", e)
else:
    print("OPENAI_API_KEY não definido — defina em ENV vars")

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
        "openai_model": OPENAI_MODEL,
        "openai_key_set": bool(OPENAI_API_KEY),
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
        hist.append({"role":"user","content":user_message})
        hist.append({"role":"assistant","content":bot_text})
        session_histories[session_id] = hist[-20:]
        return jsonify({"response": bot_text})

    # Verifica chave
    if not OPENAI_API_KEY:
        return jsonify({"error":"OPENAI_API_KEY não configurada. Defina a variável OPENAI_API_KEY no Render."}), 500

    # monta mensagens para ChatCompletion
    history = session_histories.get(session_id, [])
    # append user
    history.append({"role":"user","content":user_message})

    try:
        import openai
        resp = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=history,
            max_tokens=150,
            temperature=0.7
        )
        bot_text = resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return jsonify({"error": f"OpenAI error: {str(e)}"}), 500

    # atualiza histórico (mantém últimos 20)
    history.append({"role":"assistant","content":bot_text})
    session_histories[session_id] = history[-20:]

    return jsonify({"response": bot_text})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
