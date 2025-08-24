import unittest
from unittest.mock import patch, MagicMock
import soundfile as sf
import numpy as np
import threading
import time
import queue

from simple_voice.simple_voice import Listener, CHUNK_DURATION_MS, MIN_SILENCE_DURATION_MS


class AudioSimulator:
    def __init__(self, listener, audio, min_silence_duration_ms):
        self.listener = listener
        self.audio = audio
        self.min_silence_duration_ms = min_silence_duration_ms
        self.simulation_finished = threading.Event()
        self.mock_stream_instance = MagicMock()
        self.simulator_thread = threading.Thread(
            target=self._audio_simulator,
            args=(self.listener.audio_callback,)
        )
        self.mock_stream_instance.__enter__ = self._enter
        self.mock_stream_instance.__exit__ = self._exit

    def _audio_simulator(self, callback):
        chunk_samples = self.listener.chunk_samples
        for i in range(0, len(self.audio), chunk_samples):
            chunk = self.audio[i:i + chunk_samples]
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), 'constant')
            indata = chunk.reshape(-1, 1)
            callback(indata, len(chunk), None, None)
            time.sleep(CHUNK_DURATION_MS / 1000.0)

        silence_duration_ms = self.min_silence_duration_ms + 100
        num_silence_chunks = int(silence_duration_ms / CHUNK_DURATION_MS)
        silence_chunk = np.zeros(chunk_samples).reshape(-1, 1)
        for _ in range(num_silence_chunks):
            callback(silence_chunk, len(silence_chunk), None, None)
            time.sleep(CHUNK_DURATION_MS / 1000.0)

        self.simulation_finished.set()

    def _enter(self, *args):
        self.simulator_thread.start()

    def _exit(self, *args):
        self.simulator_thread.join()

    def wait(self, timeout=15):
        return self.simulation_finished.wait(timeout=timeout)


class TestListener(unittest.TestCase):
    def setUp(self):
        self.min_silence_duration_ms = MIN_SILENCE_DURATION_MS
        self.listener = Listener()
        self.audio, self.sr = sf.read("tests/assets/sample.wav", dtype='float32')

    @patch('simple_voice.simple_voice.sd.InputStream')
    def test_transcription(self, mock_input_stream):
        """Test that to_text correctly processes an audio file and yields transcriptions."""
        
        simulator = AudioSimulator(self.listener, self.audio, self.min_silence_duration_ms)
        mock_input_stream.return_value = simulator.mock_stream_instance

        result_queue = queue.Queue()
        
        def run_generator():
            for text in self.listener.transcription():
                result_queue.put(text)

        generator_thread = threading.Thread(target=run_generator)
        generator_thread.daemon = True
        generator_thread.start()

        self.assertTrue(simulator.wait(timeout=15), "Audio simulation timed out")

        results = []
        timeout = time.time() + 5
        while len(results) < 2 and time.time() < timeout:
            try:
                results.append(result_queue.get(timeout=1))
            except queue.Empty:
                continue
        
        full_transcription = " ".join(results)
        expected_transcription = "Here is sentence number one. And here is sentence number two."

        self.assertEqual(full_transcription.strip(), expected_transcription)

    @patch('simple_voice.simple_voice.sd.InputStream')
    def test_transcription_with_callback(self, mock_input_stream):
        """Test that text method correctly applies the callback to the transcription."""
        
        simulator = AudioSimulator(self.listener, self.audio, self.min_silence_duration_ms)
        mock_input_stream.return_value = simulator.mock_stream_instance

        result_queue = queue.Queue()
        
        def text_modifier(text):
            yield text.upper()

        def run_generator():
            for text in self.listener.transcription(callback=text_modifier):
                result_queue.put(text)

        generator_thread = threading.Thread(target=run_generator)
        generator_thread.daemon = True
        generator_thread.start()

        self.assertTrue(simulator.wait(timeout=15), "Audio simulation timed out")

        results = []
        timeout = time.time() + 5
        while len(results) < 2 and time.time() < timeout:
            try:
                results.append(result_queue.get(timeout=1))
            except queue.Empty:
                continue
        
        full_transcription = " ".join(results)
        expected_transcription = "HERE IS SENTENCE NUMBER ONE. AND HERE IS SENTENCE NUMBER TWO."

        self.assertEqual(full_transcription.strip(), expected_transcription)

    @patch('simple_voice.simple_voice.sd.InputStream')
    def test_audio(self, mock_input_stream):
        """Test that audio method correctly yields audio chunks."""
        
        simulator = AudioSimulator(self.listener, self.audio, self.min_silence_duration_ms)
        mock_input_stream.return_value = simulator.mock_stream_instance

        result_queue = queue.Queue()
        
        def run_generator():
            for audio_chunk in self.listener.audio():
                result_queue.put(audio_chunk)

        generator_thread = threading.Thread(target=run_generator)
        generator_thread.daemon = True
        generator_thread.start()

        self.assertTrue(simulator.wait(timeout=15), "Audio simulation timed out")

        results = []
        timeout = time.time() + 5
        while len(results) < 2 and time.time() < timeout:
            try:
                results.append(result_queue.get(timeout=1))
            except queue.Empty:
                continue
        
        transcriptions = [self.listener.transcribe_audio(chunk) for chunk in results]
        full_transcription = " ".join(transcriptions)
        expected_transcription = "Here is sentence number one. And here is sentence number two."

        self.assertEqual(full_transcription.strip(), expected_transcription)

    @patch('simple_voice.simple_voice.sd.InputStream')
    def test_audio_with_callback(self, mock_input_stream):
        """Test that audio method correctly applies the callback to the audio."""
        
        simulator = AudioSimulator(self.listener, self.audio, self.min_silence_duration_ms)
        mock_input_stream.return_value = simulator.mock_stream_instance

        result_queue = queue.Queue()
        
        def audio_silencer(audio_array):
            yield np.zeros_like(audio_array)

        def run_generator():
            for audio_chunk in self.listener.audio(callback=audio_silencer):
                result_queue.put(audio_chunk)

        generator_thread = threading.Thread(target=run_generator)
        generator_thread.daemon = True
        generator_thread.start()

        self.assertTrue(simulator.wait(timeout=15), "Audio simulation timed out")

        results = []
        timeout = time.time() + 5
        while len(results) < 2 and time.time() < timeout:
            try:
                results.append(result_queue.get(timeout=1))
            except queue.Empty:
                continue
        
        self.assertTrue(len(results) > 0, "Did not receive any audio chunks.")
        for chunk in results:
            self.assertTrue(np.all(chunk == 0))


if __name__ == "__main__":
    unittest.main()
