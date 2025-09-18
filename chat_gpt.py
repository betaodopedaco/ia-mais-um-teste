# app.py
import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import logging

app = Flask(__name__, template_folder="templates")
CORS(app)
logging.basicConfig(level=logging.INFO)

MODEL_NAME = os.environ.get("HF_MODEL", "facebook/blenderbot_small-90M")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", 120))
DEVICE = os.environ.get("DEVICE", "cpu")  # forçar cpu no Render

# carregar transformers + torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

device = torch.device("cpu") if DEVICE == "cpu" else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
app.logger.info(f"Usando device: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

session_histories = {}

@app.route("/")
def home():
    try:
        return render_template("index.html")
    except Exception:
        return "Servidor rodando", 200

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    user_message = (data.get("message") or "").strip()
    session_id = data.get("session_id", "default")
    if not user_message:
        return jsonify({"error":"Mensagem vazia"}), 400

    history = session_histories.get(session_id, "")
    prompt = (history + "\nUser: " + user_message + "\nBot:").strip()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
                             temperature=0.7, top_k=50, top_p=0.9, no_repeat_ngram_size=3)
    bot_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    new_history = (history + f"\nUser: {user_message}\nBot: {bot_text}") if history else f"User: {user_message}\nBot: {bot_text}"
    session_histories[session_id] = new_history[-4000:]
    return jsonify({"response": bot_text})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
