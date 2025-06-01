# simple-voice

A simple voice activity detection utility.

## Setup

1. clone the repository
```shell
git clone https://github.com/ak314/simple-voice.git
cd simple-voice
```
2. install dependencies
```shell
uv sync
```

## Test VAD on recorded audio
1. (if needed) record test file
```shell
uv run record_test_file.py
```
2. test vad
```shell
uv run try_vad.py
```

## Run Listener for live audio
Adapt the callback function in `try_listener.py` to handle the audio based on your needs. The default implementation simply plays the audio.
```shell
uv run try_listener.py
```
