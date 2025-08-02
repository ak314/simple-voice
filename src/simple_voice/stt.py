import numpy as np
import tokenizers
import onnxruntime
from huggingface_hub import hf_hub_download

from . import ASSETS_DIR


class Moonshine:
    def __init__(self, model_name="tiny", model_precision="float"):
        encoder_path, decoder_path = self._get_onnx_weights(model_name, model_precision)
        self.encoder = onnxruntime.InferenceSession(encoder_path)
        self.decoder = onnxruntime.InferenceSession(decoder_path)

        if "tiny" in model_name:
            self.num_layers = 6
            self.num_key_value_heads = 8
            self.head_dim = 36
        elif "base" in model_name:
            self.num_layers = 8
            self.num_key_value_heads = 8
            self.head_dim = 52
        else:
            raise ValueError(f'Unknown model "{model_name}"')

        self.decoder_start_token_id = 1
        self.eos_token_id = 2
        
        tokenizer_file = ASSETS_DIR / "tokenizer.json"
        self.tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_file))

    def _get_onnx_weights(self, model_name, precision="float"):
        if model_name not in ["tiny", "base"]:
            raise ValueError(f'Unknown model "{model_name}"')
        repo = "UsefulSensors/moonshine"
        subfolder = f"onnx/merged/{model_name}/{precision}"
        return (
            hf_hub_download(repo, f"{x}.onnx", subfolder=subfolder)
            for x in ("encoder_model", "decoder_model_merged")
        )

    def generate(self, audio, max_len=None):
        "audio has to be a numpy array of shape [1, num_audio_samples]"
        if max_len is None:
            # max 6 tokens per second of audio
            max_len = int((audio.shape[-1] / 16_000) * 6)

        last_hidden_state = self.encoder.run(None, dict(input_values=audio))[0]
        past_key_values = {
            f"past_key_values.{i}.{a}.{b}": np.zeros(
                (0, self.num_key_value_heads, 1, self.head_dim), dtype=np.float32
            )
            for i in range(self.num_layers)
            for a in ("decoder", "encoder")
            for b in ("key", "value")
        }
        tokens = [self.decoder_start_token_id]
        input_ids = [tokens]
        for i in range(max_len):
            use_cache_branch = i > 0
            decoder_inputs = dict(
                input_ids=np.array(input_ids, dtype=np.int64),
                encoder_hidden_states=last_hidden_state,
                use_cache_branch=np.array([use_cache_branch], dtype=np.bool_),
                **past_key_values,
            )
            logits, *present_key_values = self.decoder.run(None, decoder_inputs)
            next_token = logits[0, -1].argmax().item()
            tokens.append(next_token)
            if next_token == self.eos_token_id:
                break
            # Update values for next iteration
            input_ids = [[next_token]]
            for k, v in zip(past_key_values.keys(), present_key_values):
                if not use_cache_branch or "decoder" in k:
                    past_key_values[k] = v
        return [tokens]

    def transcribe(self, audio_path):
        import soundfile as sf
        audio, sr = sf.read(audio_path, dtype="float32")
        if sr != 16000:
            raise ValueError("Audio file must have a 16kHz sample rate")
        audio = audio.reshape(1, -1)
        tokens = self.generate(audio)[0]
        return self.tokenizer.decode(tokens, skip_special_tokens=True)
