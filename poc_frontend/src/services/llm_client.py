import requests

class LLMClient:
    def __init__(self, base_url: str):
        self.base_url = base_url

    def get_reply(self, user_text: str, extra_system: str = "") -> dict:
        payload = {"user": user_text, "extra_system": extra_system}
        resp = requests.post(self.base_url, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()
