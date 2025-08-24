# simple-voice

Simple Voice is a minimalistic voice capture and activity detection library, meant for rapidly prototyping voice-controlled shell applications with **minimal dependencies** and no runtime reliance on cloud services.

Simple Voice exposes simple APIs to detect speech activity and capture raw audio data or even transcribe it into text. See below for quickstart examples.

## Getting Started

### Install simple-voice
```shell
pip install git+https://github.com/ak314/simple-voice.git
```

### Run Listener to capture speech audio data

For starters, you can capture any audio data produced by your voice on your microphone. Once you start the Listener audio loop as shown below, it will yield audio data of your speech as numpy arrays.

```python
from simple_voice import Listener


if __name__ == "__main__":
    for audio_data in Listener().audio():
        print(f"Audio data: {audio_data}")
```

### Run Listener to transcribe speech

For many applications, you may want to convert your speech into text. You can use the transcription loop below to get text transcriptions of your speech.

```python
from simple_voice import Listener


if __name__ == "__main__":
    for text in Listener().transcription():
        print(f"You said: {text}")
        # do something groundbreaking with transcribed text
```

### Use callback to check for wake word

You can use callbacks to separate simple preprocessing or filtering on the audio/transcription data from the main logic downstream.
In the example below, the Listener only yields utterances that start with "Hey Listener": 

```python
from simple_voice import Listener


def check_wake_word(utterance):
    if utterance.lower().startswith("hey listener"):
        yield utterance


if __name__ == "__main__":
    for utterance in Listener().transcription(callback=check_wake_word):
        print(f"I heard: {utterance}")
        # do something awesome only when properly woken up
```

Remember to define your callback functions as generators that yield the processed data.

### Run automated tests
```shell
uv run python -m unittest discover -s tests
```
