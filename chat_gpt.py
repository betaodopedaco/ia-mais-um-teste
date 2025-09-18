from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# Carregar modelo e tokenizer
print("Carregando modelo DialoGPT-small...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
print("Modelo carregado com sucesso!")

# Dicionário para armazenar histórico de conversa por sessão
chat_histories = {}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '')
        session_id = data.get('session_id', 'default')
        
        if not user_message:
            return jsonify({'error': 'Mensagem vazia'}), 400
        
        if session_id not in chat_histories:
            chat_histories[session_id] = None
        
        new_user_input_ids = tokenizer.encode(user_message + tokenizer.eos_token, return_tensors='pt')
        bot_input_ids = torch.cat([chat_histories[session_id], new_user_input_ids], dim=-1) if chat_histories[session_id] is not None else new_user_input_ids
        
        chat_history_ids = model.generate(
            bot_input_ids, 
            max_length=1000,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            do_sample=True,
            no_repeat_ngram_size=3
        )
        
        bot_response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        chat_histories[session_id] = chat_history_ids
        
        return jsonify({'response': bot_response})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Servidor de chat iniciando...")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
