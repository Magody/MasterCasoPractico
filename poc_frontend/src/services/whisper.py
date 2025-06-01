from openai import OpenAI

class WhisperService:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def transcribe(self, audio_path: str) -> str:
        with open(audio_path, "rb") as f:
            # whisper-1, gpt-4o-transcribe
            r = self.client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=f,
                response_format="text",
                # language="es"
            )
        return r.strip()
