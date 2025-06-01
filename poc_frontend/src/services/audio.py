import sounddevice as sd
import soundfile as sf
from src.utils.device_finder import find_device   # your existing helper

class AudioPlayer:
    def play(self, path: str, devices=("cable", "default"), blocksize=1024):
        # read the entire file into memory (float32 stereo/mono as a 2-D NumPy array)
        data, samplerate = sf.read(path, dtype="float32", always_2d=True)

        # for each requested device, fire off a non-blocking stream
        streams = []
        for name in devices:
            idx = None if name == "default" else find_device(name, kind="output")
            # `sd.OutputStream` will handle format, channels, samplerate for us
            stream = sd.OutputStream(
                device=idx,
                samplerate=samplerate,
                channels=data.shape[1],
                blocksize=blocksize,
            )
            stream.start()
            streams.append(stream)

        # write in chunks
        total_frames = data.shape[0]
        pos = 0
        while pos < total_frames:
            chunk = data[pos : pos + blocksize]
            for st in streams:
                st.write(chunk)
            pos += blocksize

        # clean up
        for st in streams:
            st.stop()
            st.close()
