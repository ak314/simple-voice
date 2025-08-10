import unittest
from unittest.mock import patch, MagicMock
import soundfile as sf
import numpy as np
import threading
import time
import queue

from simple_voice.simple_voice import Listener, CHUNK_DURATION_MS

class TestListener(unittest.TestCase):
    def setUp(self):
        self.listener = Listener()
        self.audio, self.sr = sf.read("tests/assets/sample.wav", dtype='float32')

    @patch('simple_voice.simple_voice.sd.InputStream')
    def test_to_text(self, mock_input_stream):
        """Test that to_text correctly processes an audio file and yields transcriptions."""
        
        simulation_finished = threading.Event()

        def audio_simulator(callback):
            chunk_samples = self.listener.chunk_samples
            for i in range(0, len(self.audio), chunk_samples):
                chunk = self.audio[i:i + chunk_samples]
                if len(chunk) < chunk_samples:
                    chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), 'constant')
                indata = chunk.reshape(-1, 1)
                callback(indata, len(chunk), None, None)
                # Simulate real-time by sleeping for the chunk duration
                time.sleep(CHUNK_DURATION_MS / 1000.0)
            
            # Send enough silence to trigger VAD 'end' event
            silence_duration_ms = self.listener.min_silence_duration_ms + 100
            num_silence_chunks = int(silence_duration_ms / CHUNK_DURATION_MS)
            silence_chunk = np.zeros(chunk_samples).reshape(-1, 1)
            for _ in range(num_silence_chunks):
                callback(silence_chunk, len(silence_chunk), None, None)
                time.sleep(CHUNK_DURATION_MS / 1000.0)

            simulation_finished.set()

        mock_stream_instance = MagicMock()
        simulator_thread = threading.Thread(target=audio_simulator, args=(self.listener.audio_callback,))
        
        def enter(*args):
            simulator_thread.start()
        
        def exit(*args):
            simulator_thread.join()

        mock_stream_instance.__enter__ = enter
        mock_stream_instance.__exit__ = exit
        mock_input_stream.return_value = mock_stream_instance

        result_queue = queue.Queue()
        
        def run_generator():
            for text in self.listener.text():
                result_queue.put(text)

        generator_thread = threading.Thread(target=run_generator)
        generator_thread.daemon = True
        generator_thread.start()

        # Wait for the audio simulation to finish
        self.assertTrue(simulation_finished.wait(timeout=15), "Audio simulation timed out")

        results = []
        timeout = time.time() + 5  # 5-second timeout to get results
        while len(results) < 2 and time.time() < timeout:
            try:
                results.append(result_queue.get(timeout=1))
            except queue.Empty:
                continue
        
        # The VAD should split the audio into two sentences.
        full_transcription = " ".join(results)
        expected_transcription = "Here is sentence number one. And here is sentence number two."

        self.assertEqual(full_transcription.strip(), expected_transcription)

if __name__ == "__main__":
    unittest.main()
