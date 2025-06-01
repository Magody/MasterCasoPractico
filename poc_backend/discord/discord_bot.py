# discord_bot.py
import discord
from discord.ext import commands
from chat_client import ChatClient
from tts_engine import TTSEngine

class MiraiCog(commands.Cog):
    def __init__(self, bot: commands.Bot, chat_client: ChatClient, tts_engine: TTSEngine):
        self.bot = bot
        self.chat_client = chat_client
        self.tts_engine = tts_engine

    @commands.command(name="mirai", help="Habla con Mirai en voz y texto")
    async def mirai(self, ctx, *, message: str):
        # 1) Ensure user is in a voice channel
        if not ctx.author.voice or not ctx.author.voice.channel:
            return await ctx.send("🌟 Primero debes unirte a un canal de voz.")

        # 2) Connect or reuse
        voice_client = ctx.voice_client or await ctx.author.voice.channel.connect()

        # 3) Fetch Mirai's reply
        resp = await self.chat_client.get_response(message)
        texto   = resp["texto"]
        emocion = resp["emocion"]

        # 4) Echo the text+emotion
        await ctx.send(f"**Mirai:** {texto}\n*(Emoción: {emocion})*")

        # 5) Synthesize audio
        audio_path = await self.tts_engine.synthesize(texto)

        # 6) Play via FFmpeg
        if voice_client.is_playing():
            voice_client.stop()
        voice_client.play(
            discord.FFmpegPCMAudio(audio_path),
            after=lambda err: print("Error al reproducir:", err) if err else None
        )
