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
        stt_model_name="tiny",
        stt_model_precision="float",
        sample_rate=SAMPLE_RATE,
        channels=CHANNELS,
        chunk_samples=CHUNK_SAMPLES,
        device=None,
        vad_threshold=VAD_THRESHOLD,
        min_silence_duration_ms=MIN_SILENCE_DURATION_MS,
        speech_pad_ms=SPEECH_PAD_MS,
        pre_speech_buffer_size=PRE_SPEECH_BUFFER_SIZE,
    ):
        self.stt_model_name = stt_model_name
        self.stt_model_precision = stt_model_precision

        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_samples = chunk_samples
        self.device = device
        self.vad_threshold = vad_threshold
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self.pre_speech_buffer_size = pre_speech_buffer_size

        self.vad_iterator = None
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
            self.vad_iterator = VADIterator(
                SileroVADOnnx(),
                threshold=self.vad_threshold,
                sampling_rate=self.sample_rate,
                min_silence_duration_ms=self.min_silence_duration_ms,
                speech_pad_ms=self.speech_pad_ms,
            )
            logger.info("Silero VAD initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing Silero VAD: {e}")
            exit()

    def init_stt(self):
        try:
            self.stt = Moonshine(
                model_name=self.stt_model_name,
                model_precision=self.stt_model_precision
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

    def text(self, callback=None):
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
                        sample_rate, audio_array = self.queue.get()
                        if audio_array.size > 0:
                            text = self.transcribe_audio(audio_array)
                            self.queue.task_done()
                            if callable(callback):
                                text = callback(text)
                            yield text
                    else:
                        time.sleep(0.1)
        except KeyboardInterrupt:
            logger.info("\nExiting application.")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")

    def audio(self, callable=None):
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
                        sample_rate, audio_array = self.queue.get()
                        if audio_array.size > 0:
                            self.queue.task_done()
                            if callable:
                                audio_array = callable(audio_array)
                            yield audio_array
                    else:
                        time.sleep(0.1)
        except KeyboardInterrupt:
            logger.info("\nExiting application.")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
