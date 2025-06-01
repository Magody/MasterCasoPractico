import os
import json
from pathlib import Path
from dotenv import load_dotenv

from src.services.audio import AudioPlayer
from src.services.llm_client import LLMClient
from src.services.vtuber import VTubeService

# -----------------------------------------------------------------------------
# Load environment variables (override any existing ones)
# -----------------------------------------------------------------------------
load_dotenv(".env", override=True)

# -----------------------------------------------------------------------------
# ---- HTTP / API ----
# -----------------------------------------------------------------------------
# URL of your LLM service (Flask endpoint)
LLM_URL_SERVICE = os.getenv(
    "FLASK_URL",
    "http://127.0.0.1:5000/generate"
)

# Port on which this backend itself should run
FLASK_RUN_PORT = int(
    os.getenv("FLASK_RUN_PORT", "5500")
)

# Virtual audio cable identifier
VIRTUAL_CABLE = os.getenv(
    "VIRTUAL_CABLE",
    "cable"
)

# -----------------------------------------------------------------------------
# ---- ElevenLabs TTS ----
# -----------------------------------------------------------------------------
# Required: your ElevenLabs API key
ELEVEN_API_KEY = os.environ["ELEVENLABS_API_KEY"]

# Optional: which voice to use
ELEVEN_VOICE_ID = os.getenv("ELEVEN_VOICE_ID")

# -----------------------------------------------------------------------------
# ---- Kokoro TTS ----
# -----------------------------------------------------------------------------
# Which provider to use: "eleven" or "kokoro"
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "eleven").lower()

# Kokoro-specific settings
KOKORO_LANG_CODE = os.getenv("KOKORO_LANG_CODE", "e")

KOKORO_VOICES_PATH = "./static/kokoro/voices"

# Parse a JSON array of voices, e.g. ["ef_dora","jf_tebukuro"]
_try = os.getenv("KOKORO_VOICES", "[]")
try:
    KOKORO_VOICES = json.loads(_try)
    if not isinstance(KOKORO_VOICES, list):
        raise ValueError
except (json.JSONDecodeError, ValueError):
    KOKORO_VOICES = []
    print(f"Warning: failed to parse KOKORO_VOICES from {_try!r}")

# Parse a JSON array of weights, e.g. [0.2,1.1]
_try = os.getenv("KOKORO_VOICES_WEIGHTS", "[]")
try:
    KOKORO_VOICES_WEIGHTS = json.loads(_try)
    if not isinstance(KOKORO_VOICES_WEIGHTS, list):
        raise ValueError
    # ensure all floats
    KOKORO_VOICES_WEIGHTS = [float(w) for w in KOKORO_VOICES_WEIGHTS]
except (json.JSONDecodeError, ValueError):
    KOKORO_VOICES_WEIGHTS = []
    print(f"Warning: failed to parse KOKORO_VOICES_WEIGHTS from {_try!r}")

# Speed multiplier (1.0 = normal speed)
KOKORO_SPEED = float(os.getenv("KOKORO_SPEED", "1.0"))

# Tone adjustment (-max_semitone_shift .. +max_semitone_shift)
KOKORO_TONE = float(os.getenv("KOKORO_TONE", "0.0"))

# -----------------------------------------------------------------------------
# ---- Whisper / OpenAI ----
# -----------------------------------------------------------------------------
# Required: your OpenAI API key for Whisper & other OpenAI calls
WHISPER_API_KEY = os.environ["OPENAI_API_KEY"]

# -----------------------------------------------------------------------------
# ---- Discord Bot ----
# -----------------------------------------------------------------------------
# Required: bot token
DISCORD_TOKEN = os.environ.get("DISCORD_TOKEN")

# -----------------------------------------------------------------------------
# ---- VTube Studio ----
# -----------------------------------------------------------------------------
VTS_HOST = os.getenv("VTS_HOST", "127.0.0.1")
VTS_PORT = int(os.getenv("VTS_PORT", "8001"))
VTS_PLUGIN_NAME = os.getenv("VTS_PLUGIN_NAME", "MirAI")
VTS_DEVELOPER   = os.getenv("VTS_DEVELOPER", "Magody")
TOKEN_PATH      = Path(os.getenv("TOKEN_PATH", "./token.txt"))

# -----------------------------------------------------------------------------
# ---- Cache / Temp ----
# -----------------------------------------------------------------------------
CACHE_DIR = Path(os.getenv("CACHE_DIR", "./temp"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)




llm_client   = LLMClient(base_url=LLM_URL_SERVICE)
audio_player = AudioPlayer()

try:
    vtuber       = VTubeService()
except Exception as error:
    print(f"VTuberStudio error: {error}")
    vtuber = None
# choose TTS provider based on config
if TTS_PROVIDER == "kokoro":
    
    from src.services.tts.tts_kokoro import KokoroTTSProvider
    tts = KokoroTTSProvider()
    speech_tone = +0
else:
    # extra
    from src.services.tts.tts_eleven import ElevenLabsTTSProvider
    tts = ElevenLabsTTSProvider()
    speech_tone = +1