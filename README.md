# simple-voice

Simple Voice is a minimalistic voice capture and activity detection library, meant for rapidly prototyping voice-controlled shell applications with **minimal dependencies** and no runtime reliance on cloud services.

Simple Voice exposes simple APIs to detect speech activity and capture raw audio data or even transcribe it into text. See below for quickstart examples.

## Getting Started

### Install simple-voice
```shell
pip install git+https://github.com/ak314/simple-voice.git
```

### Run Listener to capture speech audio data
```python
from simple_voice import Listener


if __name__ == "__main__":
    for audio_data in Listener().audio():
        print(f"Audio data: {audio_data}")
```

### Run Listener to transcribe speech
```python
from simple_voice import Listener


if __name__ == "__main__":
    for text in Listener().transcription():
        print(f"You said: {text}")
```

### Run automated tests
```shell
uv run python -m unittest discover -s tests
```
