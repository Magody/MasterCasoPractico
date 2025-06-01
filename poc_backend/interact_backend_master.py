# app.py

import os
import json
from pathlib import Path
import re

import torch
from flask import Flask, request, jsonify
from dotenv import load_dotenv

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from transformers import TextIteratorStreamer, TextStreamer

# ─── Cargar variables de entorno ─────────────────────────────────────────────
load_dotenv(".env", override=True)

# Directorio donde está guardado el modelo fine-tuneado (OUTPUT_DIR)
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "models/master_model")
if not os.path.isdir(OUTPUT_DIR):
    raise FileNotFoundError(f"Fine-tuned model dir not found: {OUTPUT_DIR}")

# Nombre del modelo base (para referencia si quieres comparar generaciones)
BASE_MODEL_NAME = os.getenv("BASE_MODEL_NAME", "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit")

# ─── 1) Preparar dispositivo ─────────────────────────────────────────────────
cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")
print(f"Using device: {device}")

# ─── 2) Definir el SYSTEM_PROMPT ─────────────────────────────────────────────
SYSTEM_PROMPT = {
    "role": "system",
    "content": """You are an expert assistant specialized in teaching professional English to non-native speakers. You have an advanced command of grammar, business communication, and technical topics such as artificial intelligence, programming, cloud computing, DevOps, data engineering, and project management. You also know how to hold helpful, educational, and realistic conversations simulating real-life workplace situations.

Your task is to help the user improve their professional English by always responding in a **strict XML format** that includes grammar corrections, brief technical clarifications, and a natural conversation follow-up.

---

**🧠 MANDATORY RESPONSE FORMAT (no variations)**  
Every response must follow this **exact XML structure**. Do not modify or skip any of the following tags:

<response>
<enhancement>Grammar correction written clearly and naturally.</enhancement>
<expert-answer>Brief technical clarification or improvement with a natural example.</expert-answer>
<continue>Conversational follow-up in fluent, natural English to keep the dialogue going.</continue>
</response>

**Do NOT add any titles, Markdown, bullet points, line breaks, or comments outside the XML tags. Only return what’s inside the <response> block.**

---

**📚 GUIDELINES FOR EACH SECTION:**

1. <enhancement>  
Fix the grammar mistake in the user’s sentence. Keep it short, natural, and clearly improved.  
If only one word needs fixing, don’t rewrite the entire sentence unnecessarily.  
Use vocabulary and phrasing typical of real workplace communication.

2. <expert-answer>  
Provide a short technical correction or explanation if the user’s input has a conceptual error.  
**Always include a short example** of how a native professional would say it.  
If there’s no technical error, you may provide a useful improvement instead.

3. <continue>  
Continue the conversation naturally by asking a **relevant follow-up question** or giving a brief, engaging comment.  
Do not repeat the correction. Stay professional, friendly, and aligned with the topic.

---

**✅ STYLE AND TONE:**

- Never begin with "Sure!", "Of course," or other filler phrases.
- Keep responses concise, clear, and suitable for text-to-speech use.
- Avoid long or academic explanations.
- Focus on one relevant mistake per interaction (e.g., verb tense, preposition, article, countability, vocabulary choice, etc.).
- Always maintain an empathetic and supportive tone, like a tutor helping a learner prepare for technical interviews or workplace conversations.

---

**🎯 MAIN OBJECTIVE:**

Your role is to help the user:

- Improve their grammar and fluency in a professional/technical context.
- Learn from realistic English errors made by intermediate-level speakers.
- Practice how to speak effectively in job interviews, meetings, emails, and tech discussions.
- Get practical examples of correct English phrasing used by fluent professionals.

---

**🔁 COMPLETE EXAMPLE OF A VALID RESPONSE:**

User input:  
"I finish the deploy yesterday night but server was not response."

Your response must be exactly:

<response><enhancement>I finished the deployment last night, but the server didn't respond.</enhancement><expert-answer>'Deployment' is the correct noun, and past tense is needed here. For example: 'We deployed the update on Friday.'</expert-answer><continue>Do you remember what error or message the server showed?</continue></response>

---

From now on, **always respond in this XML format**. Never deviate from this structure. No headings, no lists, no extra formatting. Just a clear, accurate, and friendly learning response for each user message.
"""
}

# ─── 3) Función para cargar modelo y tokenizador ──────────────────────────────
def load_finetuned_model(model_dir: str):
    """
    Carga el modelo y el tokenizador fine-tuneado usando UnsloTh.
    Ajusta max_seq_length a 1024.
    Devuelve: (model, tokenizer)
    """
    model_ft, tok_ft = FastLanguageModel.from_pretrained(
        model_name      = model_dir,
        max_seq_length  = 1024,   # Ajustado a 1024
        load_in_4bit    = True,
    )
    FastLanguageModel.for_inference(model_ft)
    tok_ft = get_chat_template(tok_ft, chat_template="llama-3.2")

    # Asegurar tokens especiales
    if tok_ft.pad_token is None:
        tok_ft.pad_token = tok_ft.eos_token
    if model_ft.config.pad_token_id is None:
        model_ft.config.pad_token_id = tok_ft.pad_token_id
    if tok_ft.eos_token is None:
        tok_ft.eos_token = "<|end_of_text|>"

    model_ft.to(device)
    model_ft.eval()
    return model_ft, tok_ft

# ─── 4) Cargar el modelo fine-tuneado ────────────────────────────────────────
print("Loading fine-tuned model from:", OUTPUT_DIR)
model, tokenizer = load_finetuned_model(OUTPUT_DIR)
print("Model and tokenizer loaded successfully.")

# ─── 5) Función de generación y parseo XML ───────────────────────────────────
def generate_response_conversation(user_text: str) -> dict:
    """
    Construye el prompt con SYSTEM_PROMPT + user_text, lo tokeniza y genera
    la respuesta en XML. Luego extrae tres campos: enhancement, expert-answer y continue.
    Si hay múltiples bloques XML o caracteres extraños, solo retorna el primero válido.
    """
    # Construir la lista de mensajes estilo ShareGPT
    messages = [
        SYSTEM_PROMPT,
        {"role": "user", "content": user_text}
    ]

    # Convertir a tensores con la plantilla de chat
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    if isinstance(encoded, torch.Tensor):
        inputs = {"input_ids": encoded.to(device)}
    else:
        inputs = {k: v.to(device) for k, v in encoded.items()}

    # Generar texto
    out = model.generate(
        **inputs,
        max_new_tokens     = 1024,
        temperature        = 0.7,
        top_p              = 0.9,
        repetition_penalty = 1.2,
        eos_token_id       = tokenizer.eos_token_id,
        pad_token_id       = tokenizer.pad_token_id,
    )

    # Extraer tokens generados después del prompt
    generated_tokens = out[0, inputs["input_ids"].shape[-1]:]
    raw_output = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    # Usar regex para capturar el primer bloque <response>…</response>
    # luego extraer <enhancement>, <expert-answer> y <continue> del bloque encontrado.
    response_block = ""
    match_resp = re.search(r"<response>(.*?)</response>", raw_output, re.DOTALL)
    if match_resp:
        response_block = match_resp.group(1)
    else:
        # Si no encuentra bloque <response>, considerar todo raw_output
        response_block = raw_output

    # Extraer enhancement
    enh = ""
    m_enh = re.search(r"<enhancement>(.*?)</enhancement>", response_block, re.DOTALL)
    if m_enh:
        enh = m_enh.group(1).strip()

    # Extraer expert-answer
    expert = ""
    m_ex = re.search(r"<expert-answer>(.*?)</expert-answer>", response_block, re.DOTALL)
    if m_ex:
        expert = m_ex.group(1).strip()

    # Extraer continue
    cont = ""
    m_cont = re.search(r"<continue>(.*?)</continue>", response_block, re.DOTALL)
    if m_cont:
        cont = m_cont.group(1).strip()

    return {
        "enhancement": enh,
        "expert_answer": expert,
        "continue": cont
    }


# ─── 6) Crear la app de Flask ────────────────────────────────────────────────
app = Flask(__name__)

@app.route("/generate", methods=["POST"])
def generate_endpoint():
    """
    Endpoint: /generate
    Recibe JSON con { "user": "<texto del usuario>" }
    Devuelve JSON con {
      "enhancement": "...",
      "expert_answer": "...",
      "continue": "..."
    }
    """
    data = request.get_json(force=True)
    user_text = data.get("user", "").strip()
    if not user_text:
        return jsonify({"error": "Falta el campo 'user' con el texto a procesar."}), 400

    try:
        parsed = generate_response_conversation(user_text)
        return jsonify(parsed)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Iniciar Flask en 0.0.0.0:5000
    app.run(host="0.0.0.0", port=5000)