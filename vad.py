import warnings

import numpy as np
import onnxruntime


# The code below is adapted from https://github.com/snakers4/silero-vad.


class SileroVADOnnx():
    def __init__(self, path):
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1

        self.session = onnxruntime.InferenceSession(
            path, 
            providers=['CPUExecutionProvider'], 
            sess_options=opts,
        )

        self.reset_states()
        if '16k' in path:
            warnings.warn('This model support only 16000 sampling rate!')
            self.sample_rates = [16000]
        else:
            self.sample_rates = [8000, 16000]

    def _validate_input(self, x, sr: int):
        if x.ndim == 1:
            x = x[np.newaxis, :]
        if x.ndim > 2:
            raise ValueError(f"Too many dimensions for input audio chunk {x.ndim}")

        if sr != 16000 and (sr % 16000 == 0):
            step = sr // 16000
            x = x[:,::step]
            sr = 16000

        if sr not in self.sample_rates:
            raise ValueError(f"Supported sampling rates: {self.sample_rates} (or multiply of 16000)")
        if sr / x.shape[1] > 31.25:
            raise ValueError("Input audio chunk is too short")

        return x, sr

    def reset_states(self, batch_size=1):
        self._state = np.zeros((2, batch_size, 128), dtype=np.float32)
        self._context = np.zeros(0, dtype=np.float32)
        self._last_sr = 0
        self._last_batch_size = 0

    def __call__(self, x, sr: int):
        if isinstance(x, list):
            x = np.array(x, dtype=np.float32)
            
        x, sr = self._validate_input(x, sr)
        num_samples = 512 if sr == 16000 else 256

        if x.shape[-1] != num_samples:
            raise ValueError(f"Provided number of samples is {x.shape[-1]} (Supported values: 256 for 8000 sample rate, 512 for 16000)")

        batch_size = x.shape[0]
        context_size = 64 if sr == 16000 else 32

        if not self._last_batch_size:
            self.reset_states(batch_size)
        if (self._last_sr) and (self._last_sr != sr):
            self.reset_states(batch_size)
        if (self._last_batch_size) and (self._last_batch_size != batch_size):
            self.reset_states(batch_size)

        if not len(self._context):
            self._context = np.zeros((batch_size, context_size), dtype=np.float32)

        x = np.concatenate([self._context, x], axis=1)
        if sr in [8000, 16000]:
            ort_inputs = {'input': x, 'state': self._state, 'sr': np.array(sr, dtype='int64')}
            ort_outs = self.session.run(None, ort_inputs)
            out, state = ort_outs
            self._state = state
        else:
            raise ValueError()

        self._context = x[..., -context_size:]
        self._last_sr = sr
        self._last_batch_size = batch_size

        return out

    def audio_forward(self, x, sr: int):
        outs = []
        x, sr = self._validate_input(x, sr)
        self.reset_states()
        num_samples = 512 if sr == 16000 else 256

        if x.shape[1] % num_samples:
            pad_num = num_samples - (x.shape[1] % num_samples)
            x = np.pad(x, ((0, 0), (0, pad_num)), mode='constant', constant_values=0)

        for i in range(0, x.shape[1], num_samples):
            wavs_batch = x[:, i:i+num_samples]
            out_chunk = self.__call__(wavs_batch, sr)
            outs.append(out_chunk)

        stacked = np.concatenate(outs, axis=1)
        return stacked


class VADIterator:
    def __init__(self, model, threshold: float = 0.5, sampling_rate: int = 16000,
                 min_silence_duration_ms: int = 100, speech_pad_ms: int = 30):

        """
        Class for stream imitation

        Parameters
        ----------
        model: preloaded .onnx silero VAD model

        threshold: float (default - 0.5)
            Speech threshold. Silero VAD outputs speech probabilities for each audio chunk, probabilities ABOVE this value are considered as SPEECH.
            It is better to tune this parameter for each dataset separately, but "lazy" 0.5 is pretty good for most datasets.

        sampling_rate: int (default - 16000)
            Currently silero VAD models support 8000 and 16000 sample rates

        min_silence_duration_ms: int (default - 100 milliseconds)
            In the end of each speech chunk wait for min_silence_duration_ms before separating it

        speech_pad_ms: int (default - 30 milliseconds)
            Final speech chunks are padded by speech_pad_ms each side
        """

        self.model = model
        self.threshold = threshold
        self.sampling_rate = sampling_rate

        if sampling_rate not in [8000, 16000]:
            raise ValueError('VADIterator does not support sampling rates other than [8000, 16000]')

        self.min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
        self.speech_pad_samples = sampling_rate * speech_pad_ms / 1000
        self.reset_states()

    def reset_states(self):
        self.model.reset_states()
        self.triggered = False
        self.temp_end = 0
        self.current_sample = 0

    def __call__(self, x, return_seconds=False, time_resolution: int = 1):
        if not isinstance(x, np.ndarray):
            try:
                x = np.array(x, dtype=np.float32)
            except:
                raise TypeError("Audio cannot be converted to numpy array")

        window_size_samples = x.shape[1] if x.ndim == 2 else len(x)
        self.current_sample += window_size_samples

        speech_prob = self.model(x, self.sampling_rate).item()

        if (speech_prob >= self.threshold) and self.temp_end:
            self.temp_end = 0

        if (speech_prob >= self.threshold) and not self.triggered:
            self.triggered = True
            speech_start = max(0, self.current_sample - self.speech_pad_samples - window_size_samples)
            return {'start': int(speech_start) if not return_seconds else round(speech_start / self.sampling_rate, time_resolution)}

        if (speech_prob < self.threshold - 0.15) and self.triggered:
            if not self.temp_end:
                self.temp_end = self.current_sample
            if self.current_sample - self.temp_end < self.min_silence_samples:
                return None
            else:
                speech_end = self.temp_end + self.speech_pad_samples - window_size_samples
                self.temp_end = 0
                self.triggered = False
                return {'end': int(speech_end) if not return_seconds else round(speech_end / self.sampling_rate, time_resolution)}

        return None
