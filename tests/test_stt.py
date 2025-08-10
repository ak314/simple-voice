import unittest
from simple_voice.stt import Moonshine

class TestMoonshine(unittest.TestCase):
    def setUp(self):
        self.model = Moonshine()

    def test_transcribe_non_silent_audio(self):
        transcription = self.model.transcribe_file("tests/assets/sample.wav")
        self.assertEqual(transcription, "Here is sentence number one. And here is sentence number two.")


if __name__ == "__main__":
    unittest.main()
