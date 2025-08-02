# simple-voice

A simple voice activity detection utility.

## Getting Started

### Install simple-voice
```shell
pip install git+https://github.com/ak314/simple-voice.git
```

### Run Listener for live audio
Use the snippet below. Adapt the callback function to handle the audio based on your needs. The default implementation simply plays the audio.
```python
import sounddevice as sd

from simple_voice import Listener


def external_replay_audio(sample_rate, audio_array):
    if audio_array.size == 0:
        print("Nothing to replay.")
        return
    try:
        sd.play(audio_array, samplerate=sample_rate, blocking=True)
    except Exception as e:
        print(f"Error during playback: {e}")


if __name__ == "__main__":
    listener = Listener(processing_callback=external_replay_audio)
```

### Test VAD separately on recorded audio
1. install dependencies
```shell
pip install soundfile
```
2. (if needed) record test file
```python
import sounddevice as sd
import soundfile as sf

duration = 32/1000 * 200  # seconds
samplerate = 16000
channels = 1

print("Recording...")
audio = sd.rec(
    int(duration * samplerate), 
    samplerate=samplerate, 
    channels=channels,
    dtype='float32',
)
sd.wait()
sf.write('recorded.wav', audio, samplerate)
print("Done. Saved as recorded.wav")
```
3. test vad
```python
import soundfile as sf

from simple_voice import SileroVADOnnx, VADIterator


if __name__ == "__main__":
    vad_iterator = VADIterator(SileroVADOnnx())
    audio, sr = sf.read('recorded.wav', dtype='float32')

    for i in range(0, len(audio), 512):
        chunk = audio[i:i + 512]
        result = vad_iterator(chunk, return_seconds=True)
        if result:
            print(result)
```

### Run automated tests
```shell
uv run python -m unittest discover -s tests
```
