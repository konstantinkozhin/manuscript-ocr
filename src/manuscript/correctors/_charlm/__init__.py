"""
Character-level language model corrector for OCR post-correction.

This module provides a character-level Transformer-based MLM corrector
that fixes OCR errors by analyzing character-level patterns.
"""

import json
import os
from pathlib import Path
from typing import Optional, Union, List, Dict, Any

import numpy as np
import onnxruntime as ort

from manuscript.api.base import BaseModel
from manuscript.data import Page


class CharLM(BaseModel):
    """
    Character-level language model corrector using ONNX Runtime.

    CharLM uses a Transformer-based masked language model to correct
    OCR errors at the character level. It analyzes character confidence
    and applies corrections based on learned substitution patterns.

    Parameters
    ----------
    weights : str or Path, optional
        Path to ONNX model weights. Supports local path, URL, or preset name.
        If None, uses default preset.
    vocab : str or Path, optional
        Path to vocabulary JSON file. If None, inferred from weights location.
    substitutions : str or Path, optional
        Path to substitutions JSON file. If None, inferred from weights location.
    device : {"cuda", "cpu"}, optional
        Compute device. Default is auto-detected.
    mask_threshold : float, optional
        Confidence threshold below which characters are considered for correction.
        Default is 0.05.
    apply_threshold : float, optional
        Minimum model confidence required to apply a correction. Default is 0.95.
    max_edits : int, optional
        Maximum number of edits per word. Default is 1.
    sub_threshold : int, optional
        Minimum substitution count in training data to allow correction.
        Default is 50.
    min_word_len : int, optional
        Minimum word length to attempt correction. Default is 4.
    **kwargs
        Additional configuration options.

    Examples
    --------
    >>> from manuscript.correctors import CharLM
    >>> corrector = CharLM()
    >>> corrected_page = corrector.predict(page)
    """

    default_weights_name = None  # No default pretrained model yet
    pretrained_registry = {}

    vocab_registry = {}
    substitutions_registry = {}

    def __init__(
        self,
        weights: Optional[Union[str, Path]] = None,
        vocab: Optional[Union[str, Path]] = None,
        substitutions: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        mask_threshold: float = 0.05,
        apply_threshold: float = 0.95,
        max_edits: int = 1,
        sub_threshold: int = 50,
        min_word_len: int = 4,
        max_len: int = 32,
        **kwargs,
    ):
        # Skip BaseModel's weight resolution if weights not provided
        if weights is None:
            self.device = device or "cpu"
            self.weights = None
            self.extra_config = kwargs
            self.session = None
        else:
            super().__init__(weights=weights, device=device, **kwargs)

        self.mask_threshold = mask_threshold
        self.apply_threshold = apply_threshold
        self.max_edits = max_edits
        self.sub_threshold = sub_threshold
        self.min_word_len = min_word_len
        self.max_len = max_len

        # Vocabulary
        self.vocab_path = self._resolve_vocab(vocab) if weights else None
        self.c2i = {}
        self.i2c = {}

        # Substitutions dictionary
        self.substitutions_path = self._resolve_substitutions(substitutions) if weights else None
        self.substitutions = {}

        # ONNX session
        self.onnx_session = None

        # Load vocab and substitutions if weights provided
        if self.weights and self.vocab_path:
            self._load_vocab()
        if self.substitutions_path:
            self._load_substitutions()

    def _resolve_vocab(self, vocab: Optional[str]) -> Optional[str]:
        """Resolve vocab path, inferring from weights location if needed."""
        if vocab is not None:
            if Path(vocab).exists():
                return str(Path(vocab).absolute())
            if vocab in self.vocab_registry:
                return self._resolve_extra_artifact(
                    vocab, default_name=None, registry=self.vocab_registry, description="vocab"
                )

        # Try same directory as weights
        if self.weights:
            weights_path = Path(self.weights)
            vocab_candidate = weights_path.parent / "vocab.json"
            if vocab_candidate.exists():
                return str(vocab_candidate.absolute())

        return None

    def _resolve_substitutions(self, substitutions: Optional[str]) -> Optional[str]:
        """Resolve substitutions path, inferring from weights location if needed."""
        if substitutions is not None:
            if Path(substitutions).exists():
                return str(Path(substitutions).absolute())
            if substitutions in self.substitutions_registry:
                return self._resolve_extra_artifact(
                    substitutions, default_name=None, registry=self.substitutions_registry, description="substitutions"
                )

        # Try same directory as weights
        if self.weights:
            weights_path = Path(self.weights)
            subs_candidate = weights_path.parent / "substitutions.json"
            if subs_candidate.exists():
                return str(subs_candidate.absolute())

        return None

    def _load_vocab(self):
        """Load vocabulary from JSON file."""
        if not self.vocab_path or not Path(self.vocab_path).exists():
            return

        with open(self.vocab_path, "r", encoding="utf-8") as f:
            chars = json.load(f)

        self.c2i = {c: i for i, c in enumerate(chars)}
        self.i2c = {i: c for c, i in self.c2i.items()}

    def _load_substitutions(self):
        """Load substitutions dictionary from JSON file."""
        if not self.substitutions_path or not Path(self.substitutions_path).exists():
            return

        with open(self.substitutions_path, "r", encoding="utf-8") as f:
            self.substitutions = json.load(f)

    def _initialize_session(self):
        """Initialize ONNX Runtime session (lazy loading)."""
        if self.onnx_session is not None:
            return

        if self.weights is None:
            raise ValueError("No weights provided for CharLM corrector")

        providers = self.runtime_providers()
        self.onnx_session = ort.InferenceSession(str(self.weights), providers=providers)
        self._log_device_info(self.onnx_session)

    def predict(self, page: Page) -> Page:
        """
        Apply character-level correction to a Page.

        Parameters
        ----------
        page : Page
            Input Page object with recognized text.

        Returns
        -------
        Page
            Corrected Page object with updated word texts.
        """
        # If no weights, return copy unchanged
        if self.weights is None or not self.c2i:
            return page.model_copy(deep=True)

        # Initialize session
        if self.onnx_session is None:
            self._initialize_session()

        result = page.model_copy(deep=True)

        for block in result.blocks:
            for line in block.lines:
                for word in line.words:
                    if word.text and len(word.text) >= self.min_word_len:
                        corrected = self._correct_word(word.text)
                        word.text = corrected

        return result

    def _correct_word(self, text: str) -> str:
        """Correct a single word using the CharLM model."""
        import re
        word_pattern = re.compile(r'([а-яёіѣѵА-ЯЁІѢѴ]+)|([^а-яёіѣѵА-ЯЁІѢѴ]+)', re.IGNORECASE)

        tokens = []
        for m in word_pattern.finditer(text):
            word_part, other_part = m.groups()
            if word_part:
                tokens.append((word_part, True))
            else:
                tokens.append((other_part, False))

        result_parts = []
        for token, is_word in tokens:
            if not is_word:
                result_parts.append(token)
                continue

            word_lower = token.lower()
            if len(word_lower) < self.min_word_len:
                result_parts.append(token)
                continue

            corrected = self._correct_single_word(word_lower)
            if corrected != word_lower:
                corrected = self._restore_case(token, corrected)
            else:
                corrected = token

            result_parts.append(corrected)

        return "".join(result_parts)

    def _correct_single_word(self, word: str) -> str:
        """Apply MLM-based correction to a single lowercase word."""
        chars = list(word[:self.max_len])
        L = len(chars)
        if L == 0:
            return word

        unk = self.c2i.get("<UNK>", 2)
        mask = self.c2i.get("<MASK>", 1)
        pad = self.c2i.get("<PAD>", 0)

        # Build batch: mask each position one at a time
        batch = []
        for i in range(L):
            ids = [(self.c2i.get(ch, unk) if j != i else mask) for j, ch in enumerate(chars)]
            ids += [pad] * (self.max_len - len(ids))
            batch.append(ids)

        batch_array = np.array(batch, dtype=np.int64)

        # Run inference
        input_name = self.onnx_session.get_inputs()[0].name
        output_name = self.onnx_session.get_outputs()[0].name
        logits = self.onnx_session.run([output_name], {input_name: batch_array})[0]

        # Softmax
        probs = self._softmax(logits, axis=-1)

        # Analyze confidences
        confidences = []
        for i in range(L):
            char_id = self.c2i.get(chars[i], unk)
            p_cur = probs[i, i, char_id]
            prob_vec = probs[i, i]
            confidences.append((i, p_cur, prob_vec))

        # Find candidates below threshold
        candidates = sorted(
            [(i, p, v) for i, p, v in confidences if p < self.mask_threshold],
            key=lambda x: x[1]
        )

        # Apply corrections
        edits = 0
        for i, p_cur, prob_vec in candidates:
            if edits >= self.max_edits:
                break

            best_id = int(np.argmax(prob_vec))
            best_p = prob_vec[best_id]
            best_char = self.i2c.get(best_id, "<UNK>")

            # Skip special tokens
            if best_char in ("<UNK>", "<PAD>", "<MASK>"):
                continue
            if best_char == chars[i]:
                continue
            if best_p < self.apply_threshold:
                continue

            # Check substitution count
            sub_key = f"{chars[i]}→{best_char}"
            if self.substitutions.get(sub_key, 0) < self.sub_threshold:
                continue

            chars[i] = best_char
            edits += 1

        return "".join(chars)

    def _restore_case(self, original: str, corrected: str) -> str:
        """Restore original case pattern to corrected word."""
        result = []
        for i, ch in enumerate(corrected):
            if i < len(original) and original[i].isupper():
                result.append(ch.upper())
            else:
                result.append(ch)
        return "".join(result)

    @staticmethod
    def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Compute softmax along axis."""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    @staticmethod
    def train(
        words_path: Optional[Union[str, Path]] = None,
        text_path: Optional[Union[str, Path]] = None,
        pairs_path: Optional[Union[str, Path]] = None,
        charset_path: Optional[Union[str, Path]] = None,
        *,
        exp_dir: str = "exp_charlm",
        max_words: int = 1_500_000,
        max_len: int = 32,
        emb_size: int = 192,
        n_layers: int = 6,
        n_heads: int = 6,
        ffn_size: int = 768,
        dropout: float = 0.1,
        batch_size: int = 256,
        epochs: int = 100,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        grad_clip: float = 1.0,
        min_len: int = 5,
        mask_prob: float = 0.0,
        span_min: int = 1,
        span_max: int = 3,
        spans_min: int = 1,
        spans_max: int = 2,
        eval_ratio: float = 0.01,
        seed: int = 42,
        checkpoint: Optional[str] = None,
        **extra_config,
    ) -> str:
        """
        Train CharLM character-level language model.

        Parameters
        ----------
        words_path : str or Path, optional
            Path to words file (one word per line).
        text_path : str or Path, optional
            Path to text file for n-gram dataset.
        pairs_path : str or Path, optional
            Path to CSV file with incorrect/correct pairs.
        charset_path : str or Path
            Path to charset file (allowed characters).
        exp_dir : str, optional
            Experiment directory. Default is "exp_charlm".
        max_words : int, optional
            Maximum words to use from words file. Default is 1_500_000.
        max_len : int, optional
            Maximum sequence length. Default is 32.
        emb_size : int, optional
            Embedding size. Default is 192.
        n_layers : int, optional
            Number of transformer layers. Default is 6.
        n_heads : int, optional
            Number of attention heads. Default is 6.
        ffn_size : int, optional
            Feed-forward network size. Default is 768.
        dropout : float, optional
            Dropout rate. Default is 0.1.
        batch_size : int, optional
            Batch size. Default is 256.
        epochs : int, optional
            Number of epochs. Default is 100.
        lr : float, optional
            Learning rate. Default is 1e-3.
        weight_decay : float, optional
            Weight decay. Default is 0.01.
        grad_clip : float, optional
            Gradient clipping. Default is 1.0.
        min_len : int, optional
            Minimum word length. Default is 5.
        mask_prob : float, optional
            Probability of using span masking. Default is 0.0.
        span_min : int, optional
            Minimum span length for masking. Default is 1.
        span_max : int, optional
            Maximum span length for masking. Default is 3.
        spans_min : int, optional
            Minimum number of spans. Default is 1.
        spans_max : int, optional
            Maximum number of spans. Default is 2.
        eval_ratio : float, optional
            Evaluation set ratio. Default is 0.01.
        seed : int, optional
            Random seed. Default is 42.
        checkpoint : str, optional
            Path to checkpoint to resume from.
        **extra_config
            Additional config options.

        Returns
        -------
        str
            Path to the final checkpoint.
        """
        from .train import train as run_training
        from .config import DEFAULT_CONFIG

        config = {
            **DEFAULT_CONFIG,
            "exp_dir": exp_dir,
            "words_path": words_path,
            "text_path": text_path,
            "pairs_path": pairs_path,
            "charset_path": charset_path,
            "max_words": max_words,
            "max_len": max_len,
            "emb_size": emb_size,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "ffn_size": ffn_size,
            "dropout": dropout,
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "weight_decay": weight_decay,
            "grad_clip": grad_clip,
            "min_len": min_len,
            "mask_prob": mask_prob,
            "span_min": span_min,
            "span_max": span_max,
            "spans_min": spans_min,
            "spans_max": spans_max,
            "eval_ratio": eval_ratio,
            "seed": seed,
            "checkpoint": checkpoint,
            **extra_config,
        }

        run_training(config)

        # Return path to final checkpoint
        return os.path.join(exp_dir, "checkpoints", f"charlm_epoch_{epochs}.pt")

    @staticmethod
    def export(
        weights_path: Union[str, Path],
        vocab_path: Union[str, Path],
        output_path: Union[str, Path],
        max_len: int = 32,
        emb_size: int = 192,
        n_layers: int = 6,
        n_heads: int = 6,
        ffn_size: int = 768,
        opset_version: int = 14,
        simplify: bool = True,
    ) -> None:
        """
        Export CharLM PyTorch model to ONNX format.

        Parameters
        ----------
        weights_path : str or Path
            Path to PyTorch checkpoint (.pt file).
        vocab_path : str or Path
            Path to vocabulary JSON file.
        output_path : str or Path
            Path to save ONNX model.
        max_len : int, optional
            Maximum sequence length. Default is 32.
        emb_size : int, optional
            Embedding size. Default is 192.
        n_layers : int, optional
            Number of transformer layers. Default is 6.
        n_heads : int, optional
            Number of attention heads. Default is 6.
        ffn_size : int, optional
            Feed-forward network size. Default is 768.
        opset_version : int, optional
            ONNX opset version. Default is 14.
        simplify : bool, optional
            Apply ONNX simplification. Default is True.
        """
        import torch
        from .model import CharTransformerMLM

        weights_path = Path(weights_path)
        vocab_path = Path(vocab_path)
        output_path = Path(output_path)

        if not weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {weights_path}")
        if not vocab_path.exists():
            raise FileNotFoundError(f"Vocab not found: {vocab_path}")

        # Load vocab
        print(f"Loading vocab from {vocab_path}...")
        with open(vocab_path, "r", encoding="utf-8") as f:
            chars = json.load(f)
        vocab_size = len(chars)
        c2i = {c: i for i, c in enumerate(chars)}
        pad_idx = c2i.get("<PAD>", 0)

        print(f"Vocab size: {vocab_size}")

        # Load model
        print(f"Loading checkpoint from {weights_path}...")
        checkpoint = torch.load(str(weights_path), map_location="cpu")
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        # Create model
        model = CharTransformerMLM(
            vocab_size=vocab_size,
            emb_size=emb_size,
            max_len=max_len,
            n_layers=n_layers,
            n_heads=n_heads,
            ffn_size=ffn_size,
            dropout=0.0,  # No dropout for inference
            pad_idx=pad_idx,
        )
        model.load_state_dict(state_dict)
        model.eval()

        print(f"\n=== CharLM ONNX Export ===")
        print(f"Max length: {max_len}")
        print(f"Embedding size: {emb_size}")
        print(f"Layers: {n_layers}")
        print(f"Heads: {n_heads}")

        # Export to ONNX
        dummy_input = torch.zeros(1, max_len, dtype=torch.long)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"\nExporting to ONNX...")
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
            opset_version=opset_version,
            do_constant_folding=True,
        )

        print(f"[OK] ONNX model saved to: {output_path}")

        # Simplify if requested
        if simplify:
            try:
                import onnx
                from onnxsim import simplify as onnx_simplify

                print("Simplifying ONNX model...")
                onnx_model = onnx.load(str(output_path))
                simplified, check = onnx_simplify(onnx_model)
                if check:
                    onnx.save(simplified, str(output_path))
                    print("[OK] Model simplified successfully")
                else:
                    print("[WARN] Simplification check failed, keeping original")
            except ImportError:
                print("[SKIP] onnxsim not installed, skipping simplification")
            except Exception as e:
                print(f"[WARN] Simplification failed: {e}")

        print(f"\nExport complete: {output_path}")
