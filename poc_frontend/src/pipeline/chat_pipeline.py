import random, time
from pathlib import Path

from src.services.whisper    import WhisperService
from src import config
import datetime

from src.utils.audio_fx import _post_process
from src.config import llm_client, tts, vtuber, speech_tone

# singletons
whisper      = WhisperService(api_key=config.WHISPER_API_KEY)



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
    # LLM call
    try:
        resp       = llm_client.get_reply(user_text=prompt)
        reply_emocion = resp.get("emocion", "")
        reply_text = resp.get("texto", "")
        print(reply_emocion, reply_text)
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

def text_pipeline(user_text: str, history: list = None, func_core_pipeline_custom=None):
    """
    Exposed pipeline for text-only input.
    Si se pasa func_core_pipeline_custom, se delega en esa función en lugar de _core_pipeline.
    """
    if func_core_pipeline_custom is not None:
        return func_core_pipeline_custom(user_text=user_text, history=history)
    return _core_pipeline(user_text=user_text, history=history)




def audio_pipeline(audio_file: str, history: list = None, func_core_pipeline_custom=None):
    """
    Exposed pipeline for audio-only input.
    Primero transcribe, luego delega en _core_pipeline o en func_core_pipeline_custom si se provee.
    """
    # 0) audio → text
    user_text = whisper.transcribe(audio_file)
    if func_core_pipeline_custom is not None:
        return func_core_pipeline_custom(user_text=user_text, history=history)
    return _core_pipeline(user_text=user_text, history=history)
