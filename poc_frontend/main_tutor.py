#!/usr/bin/env python3
import datetime
import time
from pathlib import Path
import random
import sounddevice as sd
import soundfile as sf
import resampy
import numpy as np
import webrtcvad

from src.pipeline.chat_pipeline import audio_pipeline
from src import config
from src.utils.audio_fx import _post_process
from src.config import llm_client, tts, vtuber, speech_tone

# ──────────────────────────────────────────────────────────────────────────────
def find_dev(
    name_sub: str,
    kind: str,
    host_priority: tuple[str, ...] = ("MME", "Windows DirectSound", "Windows WASAPI")
) -> int:
    """
    Busca un dispositivo cuyo nombre contenga la subcadena `name_sub` (case‐insensitive)
    y que tenga canales de tipo `kind` ("input" o "output"), dando preferencia
    a los host APIs en el orden de `host_priority`.
    Retorna el índice de dispositivo que mejor coincida.
    """
    name_sub_low = name_sub.lower()
    candidates: list[tuple[int, int, str, str]] = []

    for idx, dev in enumerate(sd.query_devices()):
        dev_name = dev["name"]
        if name_sub_low in dev_name.lower() and dev[f"max_{kind}_channels"] > 0:
            api = sd.query_hostapis()[dev["hostapi"]]["name"]
            priority = host_priority.index(api) if api in host_priority else len(host_priority)
            candidates.append((priority, idx, api, dev_name))

    if not candidates:
        raise RuntimeError(f"No se encontró ningún dispositivo {kind} con '{name_sub}' en el nombre.")

    # Ordena por prioridad de hostapi, luego por índice de dispositivo
    priority, idx, api, dev_name = sorted(candidates)[0]
    print(f"→ Seleccionado {kind}: [{idx}] {dev_name} (hostapi={api})")
    return idx


# ──────────────────────────────────────────────────────────────────────────────
# Configuración
# ──────────────────────────────────────────────────────────────────────────────
AI_IN_NAME    = "Voicemeeter Out B3"   # B3: mezcla Mic + Discord
TTS_OUT_NAME  = "cable"         # la parte de *salida* del cable virtual
SR            = 48_000                 # Potato corre a 48 kHz

AI_DEVICE_ID  = find_dev(AI_IN_NAME,   kind="input")
TTS_DEVICE_ID = find_dev(TTS_OUT_NAME, kind="output")


    
print(f"→ IA escuchará desde [{AI_DEVICE_ID}] {sd.query_devices()[AI_DEVICE_ID]['name']}")
print(f"→ TTS se reproducirá en [{TTS_DEVICE_ID}] {sd.query_devices()[TTS_DEVICE_ID]['name']}")
print(f"→ Sample rate: {SR} Hz\n")

# ──────────────────────────────────────────────────────────────────────────────
class VADRecorder:
    def __init__(self, device: int, samplerate: int = SR,
                 frame_duration_ms: int = 30, silence_blocks: int = 40):
        self.vad = webrtcvad.Vad(2)  # Lower aggressiveness → fewer false-positive “silence” frames.
        self.device = device
        self.sr = samplerate
        self.frame = int(self.sr * frame_duration_ms / 1000)
        self.silence_blocks = silence_blocks

    def record_until_silence(self) -> np.ndarray:
        frames, silence, started = [], 0, False
        with sd.InputStream(device=self.device,
                            samplerate=self.sr,
                            channels=2,   # downmix interno
                            dtype="int16") as stream:
            print("🎤 IA escuchando… habla ahora.")
            while True:
                data, _ = stream.read(self.frame)
                mono = data.mean(axis=1).astype(np.int16)
                if self.vad.is_speech(mono.tobytes(), self.sr):
                    if not started:
                        print("🔊 Voz detectada, grabando…")
                        started = True
                    frames.append(mono.copy())
                    silence = 0
                else:
                    if started:
                        frames.append(mono.copy())
                        silence += 1
                        if silence > self.silence_blocks:
                            print("🔇 Silencio detectado, finalizando.")
                            break
        return np.concatenate(frames, axis=0)

# ──────────────────────────────────────────────────────────────────────────────
def play_tts(wav_path: str):
    audio, sr = sf.read(wav_path, dtype="float32")
    if sr != SR:
        audio = resampy.resample(audio.T, sr, SR).T

    sd.stop()  # limpia streams previos
    try:
        sd.play(audio, SR, device=TTS_DEVICE_ID, blocking=True)
    except sd.PortAudioError as e:
        print(f"⚠️ Error abriendo TTS [{TTS_DEVICE_ID}]: {e}")
        # aquí podrías intentar otro hostapi si lo deseas

def _core_pipeline(user_text: str, history: list = None):
    print("RECEIVED TEXT", user_text)
    history = history or []
    if not user_text:
        return history, None, history
    
    if len(history) > 10:
        history = history[-10:]

    history.append((user_text, None))
    previous = history[-3:-1]  # todos excepto el recién añadido
    history_processed = ""
    for u_text, b_text in previous:
        # sólo incluimos turnos completos (u_text + b_text)
        if b_text is None:
            continue
        history_processed += f'<MSG fuente="CHAT" usuario="danny">{u_text}</MSG>\n'
        history_processed += f'<MSG fuente="CHAT" usuario="mirai">{b_text}</MSG>\n'
    prompt = (
        # history_processed +
        f'<MSG fuente="CHAT" usuario="danny">{user_text}</MSG>'
    ).replace("\n", "").replace("\"", "'")

    print(f"PROMPT: {prompt}")
    # LLM call personalizado: llamamos al backend que devuelve tres campos
    try:
        from requests import post

        backend_resp = post(
            config.LLM_URL_SERVICE,  # e.g. "http://localhost:5000/generate"
            json={"user": prompt},
            timeout=30
        )
        backend_resp.raise_for_status()
        data = backend_resp.json()

        # Extraer enhancement, expert_answer y continue
        enhancement   = data.get("enhancement", "").strip()
        expert_answer = data.get("expert_answer", "").strip()
        cont          = data.get("continue", "").strip()

        # Construir reply_text con los prefijos requeridos para TTS
        reply_text = (
            f"English enhancement.- {enhancement} "
            f"My expert answer.- {expert_answer} "
            f"Conversation continuation.- {cont}"
        )
        print("enhancement:", enhancement)
        print("expert_answer:", expert_answer)
        print("continue:", cont)
    except Exception as e:
        reply_text = f"Error: {e}"

    history[-1] = (user_text, reply_text)

    try:
        # 1. build ./temp/YYYY-MM-DD file name
        today      = datetime.date.today().strftime("%Y-%m-%d")
        temp_dir   = Path("./temp")
        temp_dir.mkdir(parents=True, exist_ok=True)
        log_path   = temp_dir / f"history_{today}.txt"

        # 2. append the prompt with a precise timestamp
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        with log_path.open("a", encoding="utf-8") as fp:
            fp.write(f"[{ts}] {prompt}\n")      # one-liner per call

    except Exception as e:
        # don’t crash the main flow if logging fails
        print(f"⚠️  Could not write history log: {e}")

    # TTS & playback
    try:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # TODO: mkdir for mirai folder
        cache_file = Path(config.CACHE_DIR) / f"mirai/tts_{ts}.wav"

        # generate() recibirá ese temp_path y devolverá un Path
        raw_wav_path = tts.generate(reply_text, temp_path=cache_file)

        # post-process → nuevo archivo y path final
        wav_path = _post_process(Path(raw_wav_path), speech_tone)
        print("Paths: ", wav_path, raw_wav_path)

        # threading.Thread(
        #     target=audio_player.play,
        #     args=(str(wav_path), config.VIRTUAL_CABLE.split(",")),
        #     daemon=True
        # ).start()
    except Exception:
        wav_path = None

    # VTube animation
    def _animate():
        time.sleep(5)
        if vtuber:
            vtuber.trigger(random.choice([0,1,3]))

    # threading.Thread(target=_animate, daemon=True).start()

    return history, wav_path, history

# ──────────────────────────────────────────────────────────────────────────────
def main():
    history = []
    recorder = VADRecorder(device=AI_DEVICE_ID, samplerate=SR)

    iterations = 100
    while iterations > 0:
        # 1) Graba hasta silencio
        audio_np = recorder.record_until_silence()

        # 2) Guarda WAV del usuario
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # TODO: mkdir for user folder
        user_wav = Path(config.CACHE_DIR) / f"user/stt_{ts}.wav"
        sf.write(user_wav, audio_np, SR)
        print(f"💾 Usuario → {user_wav}")

        # 3) Pipeline: Whisper → LLM → TTS
        history, bot_wav_path, _ = audio_pipeline(str(user_wav), history, func_core_pipeline_custom=_core_pipeline)
        print(bot_wav_path)
        # # 4) Para pruebas rápidas, puedes desencomentar esta línea:
        # bot_wav_path = "./temp/mirai_teen.wav"

        # 5) Reproduce el TTS
        if bot_wav_path:
            print(f"▶️ Reproduciendo TTS → {bot_wav_path}")
            play_tts(bot_wav_path)
        else:
            print("⚠️ No se generó audio de respuesta.")

        # 6) Muestra la conversación
        if history:
            user_turn, bot_turn = history[-1]
            print(f"\n👤 You: {user_turn}\n🤖 Bot: {bot_turn}\n")

        iterations -= 1  # decrementa para poder parar tras X iteraciones

if __name__ == "__main__":
    main()
