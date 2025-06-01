# config.py
import os
from dotenv import load_dotenv

load_dotenv()  # reads .env in current directory

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
FLASK_URL     = os.getenv("FLASK_URL", "http://127.0.0.1:5000/generate")
