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
