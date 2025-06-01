from pathlib import Path
from typing import Union
from src.services.tts.tts_provider import TTSProvider
from src.utils.kokoro_engine import KokoroEngine

from src import config

class KokoroTTSProvider(TTSProvider):
    def __init__(self):
        # construye el motor con los parámetros de config.py
        self.engine = KokoroEngine(
            lang_code    = config.KOKORO_LANG_CODE,
            voices_path  = config.KOKORO_VOICES_PATH,      # o el path que uses para tus .pt
            default_speed= config.KOKORO_SPEED
        )
        # voces, pesos y tono desde las variables de entorno
        self.voices  = config.KOKORO_VOICES
        self.weights = config.KOKORO_VOICES_WEIGHTS
        self.tone    = config.KOKORO_TONE

    def generate(
        self,
        text: str,
        temp_path: Path = None
    ) -> Union[bytes, Path]:
        """
        Implementa el TTSProvider usando KokoroEngine.
        - Si temp_path es None, devuelve bytes WAV.
        - Si temp_path es un Path, escribe ahí el WAV y devuelve ese Path.
        """
        output_path = str(temp_path) if temp_path else None

        result = self.engine.synthesize(
            text           = text,
            voices         = self.voices,
            weights        = self.weights,
            tone           = self.tone,
            speed          = config.KOKORO_SPEED,
            split_pattern  = r'\n+',
            stream         = False,
            output_path    = output_path,
            apply_eq       = True,
            apply_vibrato  = False
        )

        # si engine devolvió una ruta (str), la envolvemos en Path
        if isinstance(result, str):
            return Path(result)
        # en caso contrario es un bytes blob de WAV
        return result
