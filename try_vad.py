import soundfile as sf

from simple_voice import SileroVADOnnx, VADIterator


if __name__ == "__main__":
    vad_model = SileroVADOnnx()

    audio, sr = sf.read('recorded.wav', dtype='float32')
    vad_iterator = VADIterator(vad_model)

    for i in range(0, len(audio), 512):
        chunk = audio[i:i + 512]
        result = vad_iterator(chunk, return_seconds=True)
        if result:
            print(result)
