import unittest
import numpy as np
import soundfile as sf
from simple_voice.stt import Moonshine
import os

class TestMoonshine(unittest.TestCase):
    def setUp(self):
        self.model = Moonshine()
        self.test_audio_path = "test_audio.wav"
        # Create a dummy silent audio file
        sr = 16000
        duration = 1  # seconds
        samples = np.zeros(int(sr * duration), dtype=np.float32)
        sf.write(self.test_audio_path, samples, sr)

    def tearDown(self):
        os.remove(self.test_audio_path)

    def test_transcribe_silent_audio(self):
        """Test that silent audio produces an empty transcription."""
        transcription = self.model.transcribe(self.test_audio_path)
        self.assertEqual(transcription.strip(), "")

    def test_transcribe_non_silent_audio(self):
        transcription = self.model.transcribe("tests/assets/sample.wav")
        self.assertEqual(transcription, "Here is sentence number one. And here is sentence number two.")


if __name__ == "__main__":
    unittest.main()
