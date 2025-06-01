#!/usr/bin/env python3
# 03_interact_frontend.py  –– Gradio + Whisper + ElevenLabs + VTubeStudio
import time
import os, random, threading, asyncio, wave
from pathlib import Path
from dotenv import load_dotenv

import requests, pyaudio, pyvts, gradio as gr
from pydub import AudioSegment
from openai import OpenAI
from gradio.themes import Default
from utils.vtuber_studio import VTubeStudioController

# ─── 0) ENV / CONFIG ───────────────────────────────────────────────────────
load_dotenv(".env", override=True)
FLASK_URL   = os.getenv("FLASK_URL", "http://127.0.0.1:5000/generate")
ELEVEN_KEY  = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID    = os.getenv("ELEVEN_VOICE_ID", "JEzse6GMhKZ6wrVNFZTq")
WHISPER_KEY = os.getenv("WHISPER_API_KEY") or os.getenv("OPENAI_API_KEY")
CACHE_DIR   = Path(os.getenv("CACHE_DIR", "./temp")); CACHE_DIR.mkdir(exist_ok=True)

if not ELEVEN_KEY or not WHISPER_KEY:
    raise RuntimeError("Missing ELEVENLABS_API_KEY or WHISPER_API_KEY in .env")

# ─── 1) ElevenLabs TTS → WAV ──────────────────────────────────────────────
def tts_to_wav(text: str, out_stem: Path) -> str:
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
    hdr = {"xi-api-key": ELEVEN_KEY, "Content-Type": "application/json"}
    payload = {
        "text": text,
        "voice_settings": {"stability": 0.7, "similarity_boost": 0.85},
        "model_id": "eleven_multilingual_v2",
        "output_format": "mp3",
    }
    mp3 = requests.post(url, json=payload, headers=hdr, timeout=60).content
    mp3_path = out_stem.with_suffix(".mp3"); mp3_path.write_bytes(mp3)
    wav_path = out_stem.with_suffix(".wav")
    AudioSegment.from_file(mp3_path, format="mp3").export(wav_path, format="wav")
    return str(wav_path)

# ─── 2) WAV playback on several devices ────────────────────────────────────
def _find_dev(pa, substr):
    s = substr.lower()
    for i in range(pa.get_device_count()):
        inf = pa.get_device_info_by_index(i)
        if inf.get("maxOutputChannels", 0) and s in inf["name"].lower():
            return i
    return None

def play_audio(path, devices=("cable", "default"), chunk=1024):
    wf = wave.open(path, "rb"); pa = pyaudio.PyAudio()
    fmt = pa.get_format_from_width(wf.getsampwidth())
    rate, ch = wf.getframerate(), wf.getnchannels()
    streams = []
    for name in devices:
        idx = None if name == "default" else _find_dev(pa, name)
        try:
            streams.append(pa.open(format=fmt, channels=ch, rate=rate,
                                   output=True, output_device_index=idx))
        except: pass
    data = wf.readframes(chunk)
    while data:
        for s in streams: s.write(data)
        data = wf.readframes(chunk)
    for s in streams: s.stop_stream(); s.close()
    wf.close(); pa.terminate()

# ─── 3) Whisper wrapper ───────────────────────────────────────────────────
class Whisperer:
    def __init__(self, key): self.cli = OpenAI(api_key=key)
    def transcribe(self, path):
        with open(path, "rb") as f:
            r = self.cli.audio.transcriptions.create(
                    model="whisper-1", file=f, response_format="text")
            return r.strip()

whisper = Whisperer(WHISPER_KEY)


# Create the controller exactly as in your working script:
vts_ctrl = VTubeStudioController(
    plugin_name="MirAI",
    developer="Magody",
    token_path="./token.txt",
    host="172.27.144.1",    # <- make sure this matches what you used successfully
    port=8001
)

# Create a dedicated event loop and connect/authenticate once at startup:
# vts_loop = asyncio.new_event_loop()
# asyncio.set_event_loop(vts_loop)
# vts_loop.run_until_complete(vts_ctrl.connect())
# vts_loop.run_until_complete(vts_ctrl.authenticate())

def trigger_only(idx, history):
    history = history or []
    vts_ctrl.trigger(idx)
    return history, None, history

# ─── 5) Backend pipeline ──────────────────────────────────────────────────
def pipeline(user_text, audio_file, history):
    history = history or []

    # 1) audio → text
    if audio_file:
        user_text = whisper.transcribe(audio_file)

    if not user_text:
        return history, None, history

    # 2) call LLM
    history.append((user_text, None))
    try:
        r = requests.post(FLASK_URL, json={"user":user_text,"extra_system":""}, timeout=60)
        r.raise_for_status()
        reply = r.json().get("reply", "<error>")
    except Exception as e:
        reply = f"Error: {e}"
    history[-1] = (user_text, reply)

    # 3) TTS
    try:
        wav = tts_to_wav(reply, CACHE_DIR / "mirai")
        threading.Thread(target=play_audio, args=(wav,), daemon=True).start()
    except: wav = None

    # 4) animation
    
    def delayed_animation():
        time.sleep(5)   # block only this helper thread
        # call the VTS trigger directly
        trigger_only(random.choice([0,1,3]), None)

    threading.Thread(target=delayed_animation, daemon=True).start()
    return history, wav, history

bright_css = """
/* ─────────── Fondo y tipografía ─────────── */
body, .gradio-container {
  background-color: #f5f7fa !important;
  color: #333 !important;
  font-family: "Segoe UI", sans-serif !important;
}
.gradio-container h1,
.gradio-container h2,
.gradio-container h3,
.gradio-container label {
  color: #1f2937 !important;
}

/* ─────────── Chatbot ─────────── */
.gr-chatbot, .gradio-chatbot {
  background-color: #ffffff !important;
  border: 1px solid #e2e8f0 !important;
  border-radius: 12px !important;
  padding: 12px !important;
}
.gr-chatbot .message.user, .gradio-chatbot .message.user {
  background-color: #e0f2f1 !important;
  color:      #004d40 !important;
  align-self: flex-end !important;
  border-radius: 12px 12px 0 12px !important;
  margin: 4px 0 !important;
}
.gr-chatbot .message.bot, .gradio-chatbot .message.bot {
  background-color: #f1f5f9 !important;
  color:      #1e293b !important;
  align-self: flex-start !important;
  border-radius: 12px 12px 12px 0 !important;
  margin: 4px 0 !important;
}

/* ─────────── Inputs y botones ─────────── */
.gr-textbox textarea,
.gr-input input,
.gr-input textarea {
  background-color: #ffffff !important;
  border: 1px solid #cbd5e1 !important;
  border-radius: 8px !important;
  padding: 8px !important;
  color: #1e293b !important;
  font-size: 16px !important;
}
.gr-button, button {
  background-color: #0284c7 !important;  /* azul claro */
  color: #ffffff !important;
  border: none !important;
  border-radius: 8px !important;
  padding: 10px 16px !important;
  box-shadow: 0 2px 6px rgba(0,0,0,0.1) !important;
  font-size: 16px !important;
  cursor: pointer !important;
}
.gr-button:hover, button:hover {
  background-color: #0369a1 !important;
}

/* ─────────── Audio uploader ─────────── */
.gr-audio, .gradio-audio {
  background-color: #ffffff !important;
  border: 1px dashed #cbd5e1 !important;
  border-radius: 8px !important;
  padding: 16px !important;
}

/* ─────────── Contenedores ─────────── */
.gr-block {
  background-color: transparent !important;
  padding: 0 !important;
}
.gr-row, .gr-column {
  gap: 8px !important;
}

.gradio-container, body {
  background-color: white !important;
}
.chatbot, .component-chatbot, .gradio-chatbot {
  background-color: white !important;
  color: black     !important;
}
.chatbot .message.user {
  background-color: #e0f7fa !important;
  color: #004d40 !important;
}
.chatbot .message.bot {
  background-color: #f1f8e9 !important;
  color: #1b5e20 !important;
}
/* ─── 1) Variables de color ──────────────────────────────────────────── */
:root {
  --bg-app: #ffffff;
  --bg-block: #ffffff;
  --bg-chat-user: #e8f5e9;    /* verde muy suave */
  --bg-chat-bot:  #e3f2fd;    /* azul muy suave  */
  --text-primary: #0f172a;    /* azul oscuro */
  --border-color: #cbd5e1;    /* gris claro */
  --btn-primary: #0284c7;     /* azul Gradio original */
  --btn-secondary: #16a34a;   /* verde Gradio */
}

/* ─── 2) Fondo general y contenedores ──────────────────────────────── */
body,
.gradio-container,
main,
.wrap,
.contain,
.column,
.block,
.wrapper,
.form,
.audio-container {
  background-color: var(--bg-app) !important;
  color:            var(--text-primary) !important;
  border-color:     var(--border-color) !important;
}

/* ─── 3) Chatbot ────────────────────────────────────────────────────── */
.bubble-wrap,
.bubble-wrap .placeholder-content {
  background-color: var(--bg-block) !important;
}
.bubble-wrap .message.user {
  background-color: var(--bg-chat-user) !important;
  color:            var(--text-primary)   !important;
  border-radius:    12px 12px 0 12px       !important;
}
.bubble-wrap .message.bot {
  background-color: var(--bg-chat-bot) !important;
  color:            var(--text-primary) !important;
  border-radius:    12px 12px 12px 0    !important;
}

/* ─── 4) Inputs y textareas ───────────────────────────────────────── */
textarea,
input,
.gr-textbox textarea,
.gr-input input {
  background-color: #ffffff      !important;
  color:            var(--text-primary) !important;
  border:           1px solid var(--border-color) !important;
  border-radius:    8px                      !important;
  padding:          8px                      !important;
}

/* ─── 5) Botones ─────────────────────────────────────────────────── */
button,
.gr-button,
.lg {
  background-color: var(--btn-primary)   !important;
  color:            #ffffff              !important;
  border:           none                 !important;
  border-radius:    8px                  !important;
  box-shadow:       0 2px 6px rgba(0,0,0,0.1) !important;
}
button.secondary,
.lg.secondary {
  background-color: var(--btn-secondary) !important;
}

/* ─── 6) Area de subida de audio ─────────────────────────────────── */
.audio-container,
.audio-container .wrap {
  background-color: #ffffff      !important;
  border:           2px dashed var(--border-color) !important;
  border-radius:    8px          !important;
  padding:          16px         !important;
}

"""


with gr.Blocks(theme=Default(), css=bright_css) as demo:
    gr.Markdown("### MirAI Chat (Gradio)")
    # Use the default Chatbot (tuple format) so your pipeline’s [(user, bot), …] works out-of-the-box
    chatbot = gr.Chatbot(label="Conversation", height=400)

    txt      = gr.Textbox(label="Your message")
    send_btn = gr.Button("Send")
    audio_up = gr.Audio(label="Upload audio", type="filepath")
    trans_btn= gr.Button("Transcribe & Send")

    
    audio_out= gr.Audio(label="Mirai (TTS playback)", type="filepath")
    state    = gr.State([])

    # click for text message
    send_btn.click(
        fn=lambda t,h: pipeline(t, None, h),
        inputs=[txt, state],
        outputs=[chatbot, audio_out, state]
    )

    # click for audio + (optional) text context
    trans_btn.click(
        fn=lambda t,f,h: pipeline(t, f, h),
        inputs=[txt, audio_up, state],
        outputs=[chatbot, audio_out, state]
    )

    # ─── NEW: Hotkey test buttons ────────────────────
    with gr.Row():
        hot0 = gr.Button("Trigger Hotkey 0") 
        hot1 = gr.Button("Trigger Hotkey 1")
        hot2 = gr.Button("Trigger Hotkey last")
    
    def trigger_and_noop(index, history):
        history = history or []
        trigger_only(index, None)
        return history, None, history

    # wire them up, re-using the same outputs as pipeline:
    hot0.click(
        fn=lambda h: trigger_and_noop(0, h),
        inputs=[state],
        outputs=[chatbot, audio_out, state],
    )
    hot1.click(
        fn=lambda h: trigger_and_noop(1, h),
        inputs=[state],
        outputs=[chatbot, audio_out, state],
    )
    hot2.click(
        fn=lambda h: trigger_and_noop(3, h),
        inputs=[state],
        outputs=[chatbot, audio_out, state],
    )

play_audio('./temp/mirai.wav')
demo.launch(share=False)
