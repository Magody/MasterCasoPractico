# main.py
import asyncio
import discord
from discord.ext import commands
from config import DISCORD_TOKEN, FLASK_URL
from chat_client import ChatClient
from tts_engine import GTTSAsyncEngine
from discord_bot import MiraiCog

intents = discord.Intents.default()
intents.message_content = True
intents.voice_states   = True

bot = commands.Bot(command_prefix="!", intents=intents)

async def main():
    # Instantiate shared components
    chat_client = ChatClient(FLASK_URL)
    tts_engine  = GTTSAsyncEngine()

    # Load our Mirai cog
    await bot.add_cog(MiraiCog(bot, chat_client, tts_engine))

    # When bot closes, also close HTTP session
    @bot.event
    async def on_close():
        await chat_client.close()

    # Start the bot
    await bot.start(DISCORD_TOKEN)

if __name__ == "__main__":
    asyncio.run(main())
