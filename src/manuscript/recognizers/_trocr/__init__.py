"""ONNX encoder/decoder OCR with local tokenizer and autoregressive generation."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from manuscript.api.recognizer import BaseRecognizer
from manuscript.utils import read_image
from .._common.region_types import RecognitionPrediction


def _resize_antialiased(image, height, width, resample, uint8_rounding=False):
    """Reproduce torchvision AA resize, including its native uint8 rounding.

    The integer kernel quantizes coefficients and clips after each axis;
    the float kernel clips and rounds only after both passes.
    """
    dtype = np.float64 if uint8_rounding else np.float32
    value = image.astype(dtype)
    for axis, size in ((1, width), (0, height)):
        original = value.shape[axis]
        if original == size:
            continue
        ratio = original / size
        scale = max(1.0, ratio)
        support = (2 if resample == 3 else 1) * scale
        coefficients = []
        for i in range(size):
            center = (i + 0.5) * ratio
            indices = np.arange(max(0, int(center - support + 0.5)),
                                min(original, int(center + support + 0.5)))
            distance = np.abs((indices + 0.5 - center) / scale).astype(dtype)
            if resample == 3:
                weights = np.where(distance < 1, ((1.5 * distance - 2.5) * distance) * distance + 1,
                                   np.where(distance < 2, ((-0.5 * distance + 2.5) * distance - 4) * distance + 2, 0))
            else:
                weights = np.maximum(0, 1 - distance)
            weights /= weights.sum()
            coefficients.append((indices, weights))
        if uint8_rounding:
            maximum = max(np.abs(weights).max() for _, weights in coefficients)
            precision = int(np.floor(np.log2(32767 / maximum)))
            divisor = 2 ** precision
        shape = list(value.shape)
        shape[axis] = size
        resized = np.empty(shape, dtype)
        for i, (indices, weights) in enumerate(coefficients):
            if uint8_rounding:
                weights = np.copysign(np.floor(np.abs(weights) * divisor + 0.5), weights)
            section = np.take(value, indices, axis=axis)
            target = [slice(None)] * 3
            target[axis] = i
            result = np.tensordot(section, weights, axes=(axis, 0))
            if uint8_rounding:
                result = np.clip(np.floor(result / divisor + 0.5), 0, 255)
            resized[tuple(target)] = result
        value = resized
    return np.rint(np.clip(value, 0, 255)).astype(np.uint8)


class TrOCR(BaseRecognizer):
    """Recognize Page crops using a compatible ONNX encoder/decoder bundle.

    Weights identify the encoder; config supplies decoder, tokenizer, image
    preprocessing and generation parameters. Full-prefix decoder graphs are
    supported, including conventional TrOCR and DINOv2/RoBERTa exports.
    KV-cache bundles add a decoder_with_past graph and cache_mapping. Image
    batches follow the encoder graph; legacy fixed-batch bundles remain usable.
    Runtime uses ONNX Runtime and tokenizers, without PyTorch/transformers.
    """

    registry_model_class = "TrOCR"
    default_weights_name = None
    pretrained_registry = {}
    config_registry = {}

    def __init__(
        self,
        weights=None,
        config=None,
        decoder=None,
        tokenizer=None,
        device=None,
        force_download=False,
        rotate_threshold=None,
        region_preparer="bbox",
        region_preparer_options=None,
        min_text_size=5,
        batch_size=1,
        generation=None,
        decoder_with_past=None,
        **kwargs,
    ):
        super().__init__(
            weights=weights, device=device, force_download=force_download,
            rotate_threshold=rotate_threshold, region_preparer=region_preparer,
            region_preparer_options=region_preparer_options,
            min_text_size=min_text_size, batch_size=batch_size, **kwargs
        )
        self._weights_preset = None
        self.config_path = self._resolve_config(config)
        if not self.config_path:
            raise ValueError("TrOCR requires an encoder companion JSON configuration")
        self.config = self._load_config_data(self.config_path)
        if (
            self.config.get("schema_version") != 1
            or self.config.get("algorithm") != "autoregressive"
        ):
            raise ValueError("Unsupported TrOCR configuration")
        p = self.config["preprocess"]
        if (
            p["height"] <= 0
            or p["width"] <= 0
            or len(p["mean"]) != 3
            or len(p["std"]) != 3
            or min(p["std"]) <= 0
            or not np.isfinite(p["mean"]).all() or not np.isfinite(p["std"]).all()
        ):
            raise ValueError("Invalid RGB preprocessing configuration")
        self._positive_integer(p["height"], "preprocess.height")
        self._positive_integer(p["width"], "preprocess.width")
        parent = Path(self.config_path).parent

        def artifact(role, explicit):
            if explicit is not None:
                return self._resolve_extra_artifact(
                    explicit, default_name=None, registry={}, description=role
                )
            resolved = getattr(self, "_resolved_model_artifacts", {}).get(role)
            path = Path(resolved) if resolved else parent / self.config[role]
            if not path.is_file():
                raise FileNotFoundError(path)
            return str(path)

        self.decoder_path = artifact("decoder", decoder)
        self.tokenizer_path = artifact("tokenizer", tokenizer)
        self.generation = {
            "num_beams": 1,
            "max_length": 128,
            "length_penalty": 1.0,
            "no_repeat_ngram_size": 0,
            "early_stopping": False,
            **{k: v for k, v in self.config["generation"].items() if v is not None},
            **(generation or {}),
        }
        if (
            int(self.generation["num_beams"]) < 1
            or int(self.generation["max_length"]) < 2
        ):
            raise ValueError("num_beams >= 1 and max_length >= 2 required")
        for key in ("num_beams", "max_length"):
            self._positive_integer(self.generation[key], key)
        repeat = self.generation["no_repeat_ngram_size"]
        if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 0:
            raise ValueError("no_repeat_ngram_size must be a nonnegative integer")
        if not np.isfinite(self.generation["length_penalty"]):
            raise ValueError("length_penalty must be finite")
        if self.generation["early_stopping"] not in (True, False, "never"):
            raise ValueError("early_stopping must be boolean or never")
        self.decoder_format = self.config.get("decoder_format", "full_prefix")
        if self.decoder_format not in ("full_prefix", "kv_cache"):
            raise ValueError("Unsupported decoder_format")
        self.decoder_with_past_path = artifact("decoder_with_past", decoder_with_past) if self.decoder_format == "kv_cache" else None
        self.cache_mapping = self.config.get("cache_mapping", []) if self.decoder_format == "kv_cache" else []
        if self.decoder_format == "kv_cache" and (not isinstance(self.cache_mapping, list) or not self.cache_mapping or
                any(not isinstance(entry, dict) or set(entry) != {"input", "output"}
                    or any(not isinstance(name, str) or not name for name in entry.values())
                    for entry in self.cache_mapping)):
            raise ValueError("KV-cache export requires cache_mapping input/output names")
        unsupported = {
            "do_sample": False,
            "repetition_penalty": 1.0,
            "min_length": 0,
            "num_return_sequences": 1,
            "encoder_no_repeat_ngram_size": 0,
            "max_new_tokens": None,
            "min_new_tokens": None,
            "forced_bos_token_id": None,
            "forced_eos_token_id": None,
            "num_beam_groups": 1,
        }
        for key, supported in unsupported.items():
            if self.generation.get(key, supported) != supported:
                raise ValueError("Unsupported generation parameter: " + key)
        self.onnx_session = self.decoder_session = self.decoder_with_past_session = self.tokenizer = None

    def _initialize_session(self):
        if self.onnx_session is not None:
            return
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise ImportError(
                'TrOCR requires tokenizers: pip install "manuscript-ocr[trocr]"'
            ) from exc
        encoder_session = self._create_onnx_session()
        decoder_session = self._create_onnx_session(self.decoder_path)
        tokenizer = Tokenizer.from_file(self.tokenizer_path)
        tokenizer.no_padding()
        tokenizer.no_truncation()
        if {x.name for x in decoder_session.get_inputs()} != {
            "input_ids",
            "encoder_hidden_states",
        }:
            raise ValueError(
                "Expected a full-prefix decoder with input_ids and encoder_hidden_states"
            )
        image = encoder_session.get_inputs()[0]
        if len(encoder_session.get_inputs()) != 1 or image.type not in (
            "tensor(float)",
            "tensor(float16)",
        ):
            raise ValueError("Encoder requires one float image input")
        expected = [
            None,
            3,
            self.config["preprocess"]["height"],
            self.config["preprocess"]["width"],
        ]
        if len(image.shape) != 4 or any(
            b is not None and isinstance(a, int) and a != b for a, b in zip(image.shape, expected)
        ):
            raise ValueError("Encoder shape disagrees with preprocessing configuration")
        if isinstance(image.shape[0], int) and image.shape[0] <= 0:
            raise ValueError("Encoder batch dimension must be positive")
        encoder_outputs = encoder_session.get_outputs()
        if not encoder_outputs or len(encoder_outputs[0].shape) != 3:
            raise ValueError("Encoder must emit [batch, sequence, hidden] states")
        past_session = None
        if self.decoder_format == "kv_cache":
            past_session = self._create_onnx_session(self.decoder_with_past_path)
            past_names = {entry["input"] for entry in self.cache_mapping}
            output_names = {entry["output"] for entry in self.cache_mapping}
            if len(past_names) != len(self.cache_mapping) or len(output_names) != len(self.cache_mapping):
                raise ValueError("Duplicate cache_mapping names")
            for session in (decoder_session, past_session):
                if not {"logits", *output_names} <= {x.name for x in session.get_outputs()}:
                    raise ValueError("Decoder cache outputs disagree with cache_mapping")
            names = {x.name for x in past_session.get_inputs()}
            if names - {"encoder_hidden_states"} != {"input_ids", *past_names}:
                raise ValueError("Cached decoder inputs disagree with cache_mapping")
            past_inputs = {x.name: x for x in past_session.get_inputs()}
            for session in (decoder_session, past_session):
                outputs = {x.name: x for x in session.get_outputs()}
                for entry in self.cache_mapping:
                    source, target = outputs[entry["output"]], past_inputs[entry["input"]]
                    if len(source.shape) != 4 or len(target.shape) != 4 or source.type != target.type:
                        raise ValueError("Decoder cache shapes/dtypes are incompatible")
        for session in (decoder_session, past_session):
            if session is None:
                continue
            logits = session.get_outputs()[0]
            if len(logits.shape) != 3 or logits.type not in ("tensor(float)", "tensor(float16)"):
                raise ValueError("Decoder must emit [batch, sequence, vocabulary] logits")
            for entry in session.get_inputs():
                if entry.name == "input_ids" and (entry.type != "tensor(int64)" or len(entry.shape) != 2):
                    raise ValueError("Decoder input_ids must be int64")
                if entry.name != "input_ids" and entry.type not in ("tensor(float)", "tensor(float16)"):
                    raise ValueError("Decoder states must be float32/float16")
                if isinstance(entry.shape[0], int) and entry.shape[0] != 1:
                    raise ValueError("Decoder requires a dynamic batch or batch 1")
                if isinstance(entry.shape[0], int) and self.generation["num_beams"] > 1:
                    raise ValueError("Beam search requires a dynamic decoder batch")
        self.decoder_with_past_session = past_session
        self.onnx_session = encoder_session
        self.decoder_session = decoder_session
        self.tokenizer = tokenizer

    def _preprocess_image(self, image):
        image = read_image(image)  # package images are RGB
        p = self.config["preprocess"]
        pil = Image.fromarray(image).convert("RGB")
        if p.get("do_resize", True):
            resample = int(p.get("resample", 3))
            if p.get("resize_backend") in ("torchvision_aa", "torchvision_aa_uint8") and resample in (2, 3):
                pil = Image.fromarray(
                    _resize_antialiased(
                        np.asarray(pil), p["height"], p["width"], resample,
                        uint8_rounding=p.get("resize_backend") == "torchvision_aa_uint8",
                    )
                )
            else:
                pil = pil.resize((p["width"], p["height"]), resample=resample)
        value = np.asarray(pil, dtype=np.float32)
        value *= float(p.get("rescale_factor", 1 / 255.0))
        value = (value - np.asarray(p["mean"], np.float32)) / np.asarray(
            p["std"], np.float32
        )
        dtype = (
            np.float16
            if self.onnx_session.get_inputs()[0].type == "tensor(float16)"
            else np.float32
        )
        return value.transpose(2, 0, 1)[None].astype(dtype)

    @staticmethod
    def _banned(tokens, n):
        if n <= 0 or len(tokens) + 1 < n:
            return []
        prefix = tuple(tokens[-(n - 1) :]) if n > 1 else ()
        return [
            tokens[i + n - 1]
            for i in range(len(tokens) - n + 1)
            if tuple(tokens[i : i + n - 1]) == prefix
        ]

    def _decode_step(self, ids, hidden, cache):
        session = self.decoder_session if cache is None else self.decoder_with_past_session
        feed = {"input_ids": ids if cache is None else ids[:, -1:]}
        inputs = {x.name: x for x in session.get_inputs()}
        if "encoder_hidden_states" in inputs:
            dtype = np.float16 if inputs["encoder_hidden_states"].type == "tensor(float16)" else np.float32
            feed["encoder_hidden_states"] = hidden.astype(dtype, copy=False)
        if cache is not None:
            feed.update(cache)
        outputs = session.run(None, feed)
        mapping = getattr(self, "cache_mapping", [])
        if not mapping:
            return outputs[0], None
        values = dict(zip((x.name for x in session.get_outputs()), outputs))
        return values["logits"], {entry["input"]: values[entry["output"]] for entry in mapping}

    def _generate_batch(self, hidden):
        from .generation import generate_batch
        return generate_batch(hidden, self.generation, self._decode_step, self._banned)

    def _generate(self, hidden):
        """Compatibility wrapper for a single encoded image."""
        return self._generate_batch(hidden)[0]

    def _predict_text_images(self, regions, batch_size=None, return_raw=False):
        if not regions:
            return []
        self._initialize_session()
        requested = self._positive_integer(self.batch_size if batch_size is None else batch_size, "batch_size")
        image_input = self.onnx_session.get_inputs()[0]
        fixed = image_input.shape[0]
        effective = fixed if isinstance(fixed, int) and fixed > 0 else requested
        # Old decoder graphs with a fixed batch cannot process several images.
        sessions = [self.decoder_session]
        if self.decoder_with_past_session is not None:
            sessions.append(self.decoder_with_past_session)
        if any(isinstance(x.shape[0], int) for session in sessions for x in session.get_inputs()):
            effective = 1
        results = []
        for offset in range(0, len(regions), effective):
            group = regions[offset:offset + effective]
            pixels = np.concatenate([self._preprocess_image(region.image) for region in group])
            if isinstance(fixed, int) and len(group) < fixed:
                pixels = np.concatenate([pixels, np.repeat(pixels[-1:], fixed - len(group), axis=0)])
            hidden = self.onnx_session.run(None, {image_input.name: pixels})[0][:len(group)]
            for index, (tokens, confidence) in enumerate(self._generate_batch(hidden)):
                results.append(RecognitionPrediction(
                    self.tokenizer.decode(tokens, skip_special_tokens=True), confidence,
                    {**({"encoder_output": hidden[index:index + 1].copy()} if return_raw else {}),
                     "token_ids": tokens, "confidence_kind": "geometric_mean_token_probability"},
                ))
        return results

    @staticmethod
    def export(source, output, *, use_cache=True, fp16=True):
        """Export a compatible source model to an ONNX bundle with configuration.

        Image batches are dynamic. use_cache=True exports initial and cached
        decoder graphs; False exports a full-prefix decoder. fp16=True also
        creates FP16 graphs, including image/state/cache inputs and outputs.
        The source is a local HF model directory. Export dependencies are
        imported only when this method is called.
        """
        from .export import export

        return export(source, output, use_cache=use_cache, fp16=fp16)
