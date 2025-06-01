import discord
from discord.ext import commands
import aiohttp
import tempfile
from src.pipeline.chat_pipeline import text_pipeline, audio_pipeline
from src.config import DISCORD_TOKEN

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    print(f"🤖 Discord bot ready as {bot.user}")

@bot.command(name="chat")
async def chat(ctx, *, message: str):
    """
    Text-only chat: !chat your message here
    """
    history = []
    history, wav_path, updated_history = text_pipeline(
        user_text=message,
        history=history
    )

    reply = updated_history[-1][1]
    await ctx.send(reply)

@bot.command(name="speak")
async def speak(ctx):
    """
    Audio or text chat:
      • If you attach an audio file, it'll be transcribed.
      • Otherwise, you can still pass text after the command.
      e.g. !speak Hello there!
    """
    history = []

    # 1) Check for an audio attachment
    if ctx.message.attachments:
        attachment = ctx.message.attachments[0]
        # Only proceed if it looks like audio
        if attachment.content_type and attachment.content_type.startswith("audio"):
            # download to a temp file
            suffix = "." + attachment.filename.split(".")[-1]
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                await attachment.save(tmp.name)
                audio_path = tmp.name

            history, wav_path, updated_history = audio_pipeline(
                audio_file=audio_path,
                history=history
            )

            reply = updated_history[-1][1]
            await ctx.send(reply)
            return

    # 2) Fallback: treat the rest of the message as text
    #    e.g. user typed "!speak how are you?"
    content = ctx.message.content[len("!speak"):].strip()
    if content:
        history, wav_path, updated_history = text_pipeline(
            user_text=content,
            history=history
        )
        reply = updated_history[-1][1]
        await ctx.send(reply)
    else:
        # no audio, no text
        await ctx.send("❓ Please provide either an audio file or text after the command.")

def run_discord():
    bot.run(DISCORD_TOKEN)
