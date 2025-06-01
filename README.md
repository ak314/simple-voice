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

## Test on recorded audio
1. (if needed) record test file
```shell
uv run record_test_file.py
```
2. test vad
```shell
uv run try_vad.py
```

## Test on live audio
```shell
uv run try_listener.py
```
