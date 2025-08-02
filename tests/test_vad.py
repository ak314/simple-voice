import unittest
import soundfile as sf
from simple_voice.vad import VADIterator, SileroVADOnnx

class TestVAD(unittest.TestCase):
    def setUp(self):
        self.vad_iterator = VADIterator(SileroVADOnnx())
        self.audio_path = "tests/assets/sample.wav"

    def test_vad_on_sample(self):
        """Test VAD on a sample audio file."""
        audio, sr = sf.read(self.audio_path, dtype='float32')

        speech_start_count = 0
        for i in range(0, len(audio), 512):
            chunk = audio[i:i + 512]
            if len(chunk) < 512:
                break 
            result = self.vad_iterator(chunk, return_seconds=True)
            if result:
                if 'start' in result:
                    speech_start_count += 1
        
        self.assertEqual(speech_start_count, 2, "Should detect 2 speech starts")

if __name__ == "__main__":
    unittest.main()
