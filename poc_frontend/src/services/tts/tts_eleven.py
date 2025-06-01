# poc_frontend/src/services/tts/tts_eleven.py
import io
import uuid
import requests
from pathlib import Path
from typing import Union
from pydub import AudioSegment

from src import config
from .tts_provider import TTSProvider

class ElevenLabsTTSProvider(TTSProvider):
    """
    TTSProvider que utiliza la API de ElevenLabs para síntesis.
    Devuelve bytes WAV o escribe un archivo WAV si se le pasa temp_path.
    """

    def __init__(self):
        self.api_key  = config.ELEVEN_API_KEY
        self.voice_id = config.ELEVEN_VOICE_ID
        self.cache_dir = config.CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.endpoint = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}"

    def generate(
        self,
        text: str,
        temp_path: Path = None
    ) -> Union[bytes, Path]:
        """
        :param text: Texto a sintetizar.
        :param temp_path: Opcional Path donde escribir el WAV resultante.
                          Si es None, devuelve bytes del WAV.
        :return: Path al archivo WAV (si temp_path) o bytes WAV.
        """
        # 1) determina ruta de salida
        if temp_path:
            wav_path = Path(temp_path)
        else:
            # nombre único basado en UUID
            stem = f"eleven_{uuid.uuid4().hex}"
            wav_path = self.cache_dir / f"{stem}.wav"

        # 2) llama a la API de ElevenLabs
        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "text": text,
            "voice_settings": {
                # puedes parametrizar estos valores vía env si quieres
                "stability": 0.7,
                "similarity_boost": 0.85
            },
            "model_id": "eleven_multilingual_v2",
            # la API por defecto devuelve MP3
            "output_format": "mp3"
        }

        resp = requests.post(
            self.endpoint,
            json=payload,
            headers=headers,
            timeout=60
        )
        resp.raise_for_status()
        mp3_data = resp.content

        # 3) convierte MP3→WAV en memoria
        mp3_buf = io.BytesIO(mp3_data)
        audio_seg = AudioSegment.from_file(mp3_buf, format="mp3")
        audio_seg.export(wav_path, format="wav")

        # 4) devuelve ruta o bytes
        if temp_path:
            return wav_path

        return wav_path.read_bytes()
