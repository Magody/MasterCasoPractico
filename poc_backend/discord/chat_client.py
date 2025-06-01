# chat_client.py
import aiohttp

class ChatClient:
    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.session = aiohttp.ClientSession()

    async def get_response(self, user_text: str, extra_system: str = "") -> dict:
        """
        Calls your Flask /generate endpoint with JSON {"user": ..., "extra_system": ...}
        and returns {"texto": "...", "emocion": "..."}.
        """
        payload = {"user": user_text, "extra_system": extra_system}
        async with self.session.post(self.endpoint, json=payload) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return {
                "texto":   data.get("texto",   ""),
                "emocion": data.get("emocion", "")
            }

    async def close(self):
        await self.session.close()
