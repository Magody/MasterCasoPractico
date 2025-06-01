# app.py
import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from unsloth import FastLanguageModel

# ─── 1) Load env vars & model once ───────────────────────────────────────
load_dotenv(".env", override=True)
MODEL_DIR = os.getenv("MODEL_DIR", "./models/Llama-3_2-3B-bnb-4bit-2025-03-22-20-epochs-overfit")
if not os.path.isdir(MODEL_DIR):
    raise FileNotFoundError(f"Model dir not found: {MODEL_DIR}")

# Load model + tokenizer in 4-bit, with LoRA adapters already merged
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = MODEL_DIR,
    max_seq_length = 512,
    load_in_4bit   = True,
    dtype          = None,
)
FastLanguageModel.for_inference(model)
device = next(model.parameters()).device
print(f"Loaded model on {device}")

# Prebuild prompt template
BOS = tokenizer.bos_token or ""
CHAT_TEMPLATE = (
    f"{BOS}Eres una streamer divertida, carismática y auténtica. Usa humor en tus respuestas.\n\n"
    "## Información adicional\n"
    "{EXTRA_SYSTEM}\n"
    "### Chat\n"
    "{CHAT}\n"
    "## Tu respuesta\n"
)

def generate_reply(user_text: str, extra_system: str = "") -> str:
    # 1) build prompt
    prompt = CHAT_TEMPLATE.format(
        EXTRA_SYSTEM = extra_system,
        CHAT         = f"[danny]: {user_text}"
    )

    # 2) tokenize (no padding so PAD≠EOS)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
        padding=False
    ).to(device)

    # 3) generate
    out = model.generate(
        **inputs,
        max_new_tokens     = 64,
        temperature        = 0.7,
        top_p              = 0.9,
        repetition_penalty = 1.1,
        eos_token_id       = tokenizer.eos_token_id,
        pad_token_id       = tokenizer.pad_token_id,
    )

    # 4) decode & split off the answer
    full = tokenizer.decode(out[0], skip_special_tokens=True)
    return full.split("## Tu respuesta")[-1].strip()

# ─── 2) Flask app ────────────────────────────────────────────────────────
app = Flask(__name__)

@app.route("/generate", methods=["POST"])
def generate_endpoint():
    data = request.get_json(force=True)
    user = data.get("user", "").strip()
    extra = data.get("extra_system", "").strip()
    if not user:
        return jsonify({"error": "Missing 'user' field"}), 400

    try:
        reply = generate_reply(user, extra)
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # listen on all interfaces so you can call it from Windows
    app.run(host="0.0.0.0", port=5000)
