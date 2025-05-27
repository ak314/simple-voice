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
