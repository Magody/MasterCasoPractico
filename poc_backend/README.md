# 1. Crear y activar el entorno con Python 3.11
conda create -n env_vtuber python=3.11 -y
conda activate env_vtuber

# 2. Instalar conda-forge packages requeridos (audio, FFmpeg, dotenv)
conda install -c conda-forge pyaudio ffmpeg python-dotenv -y

# 3. Instalar el resto de dependencias vía pip
python -m pip install requests pyvts gradio pydub openai
python -m pip install discord.py aiohttp python-dotenv gTTS
