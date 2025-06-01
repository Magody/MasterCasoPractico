# tts_engine.py
import abc
import asyncio
import uuid
import os
from gtts import gTTS

class TTSEngine(abc.ABC):
    @abc.abstractmethod
    async def synthesize(self, text: str) -> str:
        """
        Generate an audio file from `text`.
        Returns the filepath to the generated audio (e.g. an .mp3 or .wav).
        """
        ...

class GTTSAsyncEngine(TTSEngine):
    def __init__(self, output_dir: str = "tts_output"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    async def synthesize(self, text: str) -> str:
        # run blocking I/O in executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_synthesize, text)

    def _sync_synthesize(self, text: str) -> str:
        filename = f"{uuid.uuid4()}.mp3"
        path = os.path.join(self.output_dir, filename)
        tts = gTTS(text, lang="es")
        tts.save(path)
        return path
