🇪🇸 'e' => Spanish es
🇫🇷 'f' => French fr-fr
🇮🇳 'h' => Hindi hi
🇮🇹 'i' => Italian it
🇧🇷 'p' => Brazilian Portuguese pt-br

https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md#spanish

https://github.com/hexgrad/kokoro




INPUT:
	Hardware:
		- Input 1: Microphone -> Headphones (A2), Streaming (B1), VoiceChatExt (B2), VoiceChatVtuber (B3)
		- Input 2: Virtual Cable Output -> Speakers (A1), Headphones (A2), Streaming (B1), VoiceChatExt (B2)
		- Input 3: NULL
		- Input 4: NULL
		- Input 5: NULL
	Virtual:
		- Voicemeeter input: Windows -> Speakers (A1), Headphones (A2), Streamimg (B1), VoiceChatVtuber (B3)
		- Voicemeeter AUX I: VoiceChatExt (Discord) -> Speakers (A1), Headphones (A2), Streamimg (B1), VoiceChatVtuber (B3)

OUTPUT:
	Hardware:
		- A1: Speakers
		- A2: Headphones
		- A3: NULL
		- A4: NULL
		- A5: NULL
	Virtual:
		- Voicemeeter B1: Streaming (OBS)
		- Voicemeeter B2: VoiceChatExt (my microphone and vtuber audio)
		- Voicemeeter B3: VoiceChatVtuber (my microphone and discord audio)
		

	
Direct software mapping:
	- Discord input: Voicemeeter Out B2
	- Discord output: Voicemeeter Aux I

Virtual Cable:
	- Input: VTUBER Studio
	- Output: VTUBER Studio