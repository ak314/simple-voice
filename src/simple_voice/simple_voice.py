import sounddevice as sd
import numpy as np
import time
import queue
import threading
import logging

from .vad import SileroVADOnnx, VADIterator
from .stt import Moonshine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# --- Configuration ---
SAMPLE_RATE = 16000  # Silero VAD expects 16kHz
CHANNELS = 1
CHUNK_DURATION_MS = 32  # Duration of each audio chunk processed by VAD (milliseconds)
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)

# VAD Parameters (tune these to your needs)
VAD_THRESHOLD = 0.5               # Speech probability threshold
MIN_SILENCE_DURATION_MS = 700     # How long silence must be to be considered a "pause"
SPEECH_PAD_MS = 30                # Add padding to the start/end of detected speech
PRE_SPEECH_BUFFER_SIZE = 15       # Number of chunks to buffer before speech starts


class Listener:
    def __init__(
        self,
        stt_model: Moonshine = None,
        vad_iterator: VADIterator = None,
        sample_rate=SAMPLE_RATE,
        channels=CHANNELS,
        chunk_samples=CHUNK_SAMPLES,
        device=None,
        pre_speech_buffer_size=PRE_SPEECH_BUFFER_SIZE,
    ):
        self.stt = stt_model
        self.vad_iterator = vad_iterator

        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_samples = chunk_samples
        self.device = device
        self.pre_speech_buffer_size = pre_speech_buffer_size

        self.is_capturing_speech = False
        self.frames_for_processing = []
        self.queue = queue.Queue()

        self.init_vad()
        self.init_stt()

    def audio_callback(self, indata, frames, callback_time, status):
        if status:
            logger.warning(f"Sounddevice status: {status}")

        if self.vad_iterator is None:
            logger.error("VAD not initialized yet.")
            return

        try:
            speech_dict = self.vad_iterator(indata[:, 0].astype(np.float32), return_seconds=False)
        except Exception as e:
            logger.error(f"Error calling VAD: {e}")
            if self.vad_iterator:
                self.vad_iterator.reset_states()
            return

        # Always append the current frame
        self.frames_for_processing.append(indata.copy())

        if speech_dict:
            if "start" in speech_dict:
                if not self.is_capturing_speech:
                    logger.info("Speech started.")
                    self.is_capturing_speech = True
            elif "end" in speech_dict:
                if self.is_capturing_speech:
                    logger.info("Speech ended (pause detected).")
                    if self.frames_for_processing:
                        self.queue.put((self.sample_rate, np.concatenate(self.frames_for_processing)))
                    self.frames_for_processing = []
                    self.is_capturing_speech = False
                self.vad_iterator.reset_states()

        # Maintain buffer size when not capturing speech
        if not self.is_capturing_speech:
            while len(self.frames_for_processing) > self.pre_speech_buffer_size:
                self.frames_for_processing.pop(0)

    def init_vad(self):
        try:
            if self.vad_iterator:
                logger.info("Using provided VAD iterator.")
            else:
                logger.info("Initializing default VAD iterator.")
                self.vad_iterator = VADIterator(
                    SileroVADOnnx(),
                    threshold=VAD_THRESHOLD,
                    sampling_rate=self.sample_rate,
                    min_silence_duration_ms=MIN_SILENCE_DURATION_MS,
                    speech_pad_ms=SPEECH_PAD_MS,
                )
            logger.info("Silero VAD initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing Silero VAD: {e}")
            exit()

    def init_stt(self):
        try:
            if self.stt:
                logger.info("Using provided STT model.")
            else:
                logger.info("Initializing default Moonshine STT model.")
                self.stt = Moonshine(
                    model_name="tiny",
                    model_precision="float"
                )
            logger.info("Moonshine STT initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing Moonshine STT: {e}")
            exit()

    def transcribe_audio(self, audio_array: np.ndarray) -> str:
        logger.info("Transcribing audio...")
        audio_array = audio_array.astype(np.float32)
        text = self.stt.transcribe_audio(audio_array)
        logger.info(f"Transcription: {text}")
        return text

    def _listen_and_process(self, process_item, callback=None):
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                blocksize=self.chunk_samples,
                callback=self.audio_callback,
                device=self.device,
            ):
                while True:
                    if not self.queue.empty():
                        _sample_rate, audio_array = self.queue.get()
                        if audio_array.size > 0:
                            processed_item = process_item(audio_array)
                            self.queue.task_done()
                            if callback:
                                processed_item = callback(processed_item)
                            yield processed_item
                    else:
                        time.sleep(0.1)
        except KeyboardInterrupt:
            logger.info("\nExiting application.")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")

    def transcription(self, callback=None):
        return self._listen_and_process(self.transcribe_audio, callback)

    def audio(self, callback=None):
        return self._listen_and_process(lambda audio_array: audio_array, callback)
