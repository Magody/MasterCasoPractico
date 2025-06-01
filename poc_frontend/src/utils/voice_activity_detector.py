import numpy as np
from src.utils.device_finder import find_device
import sounddevice as sd

class VADAudioRecorder:
    def __init__(
        self,
        vad,
        device: int | str,
        samplerate: int = 16000,
        frame_duration_ms: int = 30,
        silence_blocks: int = 20
    ):
        self.vad = vad

        if isinstance(device, str):
            device = find_device(device, kind="input",  raise_on_missing=True)
        self.device = device

        self.samplerate = samplerate
        self.frame_size = int(samplerate * frame_duration_ms / 1000)
        self.silence_blocks = silence_blocks

        # 1) figure out how many channels the device supports
        dev_info = sd.query_devices(self.device, kind='input')
        self.input_channels = dev_info['max_input_channels']
        if self.input_channels < 1:
            raise ValueError(f"Device {device} has no input channels!")

    def record_audio(self) -> np.ndarray:
        audio_frames = []
        silence_count = 0
        started = False

        with sd.InputStream(
            device=self.device,
            samplerate=self.samplerate,
            channels=self.input_channels,
            dtype="int16"
        ) as stream:
            print("🎤 Esperando audio… habla ahora!")

            while True:
                data, overflowed = stream.read(self.frame_size)
                # data shape: (frames, input_channels)

                # 2) downmix to mono
                if self.input_channels > 1:
                    # average across channels, then to int16
                    mono = data.mean(axis=1).astype(np.int16)
                else:
                    mono = data[:, 0]

                pcm_bytes = mono.tobytes()
                is_speech = self.vad.is_speech(pcm_bytes, self.samplerate)

                if is_speech:
                    if not started:
                        print("🔊 Audio detectado, grabando…")
                        started = True
                    audio_frames.append(mono.copy())
                    silence_count = 0
                else:
                    if started:
                        silence_count += 1
                        audio_frames.append(mono.copy())
                        if silence_count > self.silence_blocks:
                            print("🔇 Silencio detectado, terminando.")
                            break

        # 3) concatenate all mono frames into a single 1-D array
        return np.concatenate(audio_frames, axis=0)