import json
import os
import re
from pathlib import Path
from typing import Optional, Union

import numpy as np

from manuscript.api.base import managed_session_initialization
from manuscript.api.corrector import BaseCorrector
from manuscript.data import Page


class CharLM(BaseCorrector):
    """
    Character-level language model corrector using ONNX Runtime.

    CharLM uses a Transformer-based masked language model to correct
    OCR substitution errors. It estimates each character's probability from
    its masked context; recognition_confidence from OCR is not used.

    Parameters
    ----------
    weights : str or Path, optional
        Path or identifier for ONNX model weights. Supports:

        - Local file path: ``"path/to/model.onnx"``
        - HTTP/HTTPS URL: ``"https://example.com/model.onnx"``
        - GitHub release: ``"github://owner/repo/tag/file.onnx"``
        - Google Drive: ``"gdrive:FILE_ID"``
        - Preset name: ``"prereform_charlm_g1"`` or ``"modern_charlm_g1"`` (from the model registry)
        - ``None``: auto-downloads default preset (prereform_charlm_g1)

    vocab : str or Path, optional
        Path to vocabulary JSON file. Required for local weights; registry presets
        supply their own vocabulary artifact.
    lexicon : str, Path, or set, optional
        Word list for dictionary-based validation. Supports:

        - Local file path: ``"path/to/words.txt"``
        - Model registry key: ``"prereform_charlm_g1"`` or ``"modern_charlm_g1"``
          (loads its lexicon artifact)
        - Python set: ``{"word1", "word2", ...}``
        - ``None``: auto-downloads default lexicon for model preset
          (prereform_words for prereform_charlm_g1, modern_words for modern_charlm_g1)

    device : {"cuda", "coreml", "cpu"}, optional
        Compute device. If None, selects CUDA when ONNX Runtime reports GPU
        support, otherwise CPU. Select CoreML explicitly with ``device="coreml"``.
    mask_threshold : float, optional
        Confidence threshold below which characters are considered for correction.
        Default is 0.05.
    apply_threshold : float, optional
        Minimum model confidence required to apply a correction. Default is 0.95.
    max_edits : int, optional
        Maximum number of edits per word. Default is 2.
    min_word_len : int, optional
        Minimum word length to attempt correction. Default is 4.
    max_len : int, optional
        Model context length. Longer words are preserved. Default is 32.
    correction_levels : sequence of str, optional
        TextRegion levels to correct: word, line or block. Default is ("word",).
        Text is split into words while preserving case and separators.
    **kwargs
        Additional configuration options.

    Examples
    --------
    >>> from manuscript.correctors import CharLM
    >>> corrector = CharLM()
    >>> corrected_page = corrector.predict(page)
    """

    default_weights_name = "prereform_charlm_g1"
    registry_model_class = "CharLM"
    def __init__(
        self,
        weights: Optional[Union[str, Path]] = None,
        vocab: Optional[Union[str, Path]] = None,
        lexicon: Optional[Union[str, Path, set]] = None,
        device: Optional[str] = None,
        force_download: bool = False,
        mask_threshold: float = 0.05,
        apply_threshold: float = 0.95,
        max_edits: int = 2,
        min_word_len: int = 4,
        max_len: int = 32,
        correction_levels=("word",),
        **kwargs,
    ):
        self._configure_correction_levels(correction_levels)

        if weights is None and self.default_weights_name is not None:
            weights = self.default_weights_name

        if weights is None:
            self.device = device or "cpu"
            self.force_download = bool(force_download)
            self.weights = None
            self.extra_config = kwargs
            self.session = None
        else:
            super().__init__(
                weights=weights,
                device=device,
                force_download=force_download,
                correction_levels=correction_levels,
                **kwargs,
            )

        self.mask_threshold = mask_threshold
        self.apply_threshold = apply_threshold
        self.max_edits = max_edits
        self.min_word_len = min_word_len
        self.max_len = max_len

        self._word_pattern = re.compile(r"(\w+)|(\W+)", re.UNICODE)

        self.vocab_path = self._resolve_vocab(vocab) if self.weights else None
        self.c2i = {}
        self.i2c = {}

        self.lexicon = None
        if lexicon is not None:
            if isinstance(lexicon, set):
                self.lexicon = frozenset(w.lower() for w in lexicon)
            else:
                lexicon_path = self._resolve_lexicon(lexicon)
                if lexicon_path:
                    self._load_lexicon(lexicon_path)
        elif getattr(self, '_resolved_model_name', None):
            from manuscript.models import info, resolve

            name = self._resolved_model_name
            if info(name, self.registry_model_class)['artifacts'].get('lexicon'):
                lexicon_path = resolve(name, self.registry_model_class, artifact='lexicon',
                                       force_download=self.force_download)['lexicon']
                self._load_lexicon(lexicon_path)

        self.onnx_session = None

        if self.weights and self.vocab_path:
            self._load_vocab()

    def _resolve_vocab(self, vocab: Optional[str]) -> Optional[str]:
        """Resolve an explicit vocab path or the registered model artifact."""
        if vocab is not None:
            return self._resolve_extra_artifact(
                vocab, default_name=None, registry={}, description="vocab"
            )

        resolved = getattr(self, '_resolved_model_artifacts', {}).get('vocab')
        if resolved:
            return str(resolved)

        raise ValueError("CharLM local weights require an explicit vocab path")

    def _resolve_lexicon(self, lexicon: str) -> Optional[str]:
        """Resolve lexicon path from registry or local file."""
        return self._resolve_extra_artifact(
            lexicon, default_name=None, registry={}, description="lexicon"
        )

    def _load_vocab(self):
        """Load vocabulary from JSON file."""
        if not self.vocab_path or not Path(self.vocab_path).exists():
            return

        with open(self.vocab_path, "r", encoding="utf-8") as f:
            chars = json.load(f)

        self.c2i = {c: i for i, c in enumerate(chars)}
        self.i2c = {i: c for c, i in self.c2i.items()}

    def _load_lexicon(self, lexicon_path: str):
        """Load lexicon (word list) from text file."""
        if not Path(lexicon_path).exists():
            return

        with open(lexicon_path, "r", encoding="utf-8") as f:
            words = set(line.strip().lower() for line in f if line.strip())
        self.lexicon = frozenset(words)

    @managed_session_initialization
    def _initialize_session(self):
        """Initialize ONNX Runtime session (lazy loading)."""
        if self.onnx_session is not None:
            return

        if self.weights is None:
            raise ValueError("No weights provided for CharLM corrector")

        self.onnx_session = self._create_onnx_session()

    def predict(self, page: Page, image: Optional[np.ndarray] = None) -> Page:
        """
        Apply character-level correction to a Page.

        Parameters
        ----------
        page : Page
            Input Page object with recognized text.
        image : np.ndarray, optional
            Optional source image associated with ``page``.
            Reserved for integrations that may pass the original image
            to correctors. ``CharLM`` does not use this argument.

        Returns
        -------
        Page
            Page copy with corrected text at the selected levels.
        """
        _ = image
        result = self._copy_page(page)

        if self.weights is None or not self.c2i:
            return result

        regions = list(self._text_regions(result))
        if not regions:
            return result

        if self.onnx_session is None:
            self._initialize_session()

        for region in regions:
            region.text = self._correct_word(region.text)

        return result

    def _correct_word(self, text: str) -> str:
        """Correct a single word using the CharLM model."""
        tokens = []
        for m in self._word_pattern.finditer(text):
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

            if self.lexicon and word_lower in self.lexicon:
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
        # Words outside the model context must remain intact.
        if len(word) > self.max_len:
            return word
        chars = list(word)
        L = len(chars)
        if L == 0:
            return word

        unk = self.c2i.get("<UNK>", 0)
        mask = self.c2i.get("<MASK>", 1)
        pad = self.c2i.get("<PAD>", 0)

        batch = []
        for i in range(L):
            ids = [
                (self.c2i.get(ch, unk) if j != i else mask)
                for j, ch in enumerate(chars)
            ]
            ids += [pad] * (self.max_len - len(ids))
            batch.append(ids)

        batch_array = np.array(batch, dtype=np.int64)

        input_name = self.onnx_session.get_inputs()[0].name
        output_name = self.onnx_session.get_outputs()[0].name
        try:
            logits = self.onnx_session.run([output_name], {input_name: batch_array})[0]
        except Exception as e:
            import warnings

            warnings.warn(
                f"Corrector inference error for word '{word}': {e}. "
                "Returning original text. This may indicate a vocab/weights mismatch.",
                RuntimeWarning,
            )
            return word

        probs = self._softmax(logits, axis=-1)
        vocab_size = probs.shape[-1]

        # Preserve characters the model vocabulary cannot represent.
        unknown_positions = set()
        for i, ch in enumerate(chars):
            if ch not in self.c2i or self.c2i[ch] >= vocab_size:
                unknown_positions.add(i)

        confidences = []
        for i in range(L):
            if i in unknown_positions:
                continue
            char_id = self.c2i[chars[i]]
            p_cur = probs[i, i, char_id]
            prob_vec = probs[i, i]
            confidences.append((i, p_cur, prob_vec))

        candidates = sorted(
            [(i, p, v) for i, p, v in confidences if p < self.mask_threshold],
            key=lambda x: x[1],
        )

        edits = 0
        for i, p_cur, prob_vec in candidates:
            if edits >= self.max_edits:
                break

            best_id = int(np.argmax(prob_vec))
            best_p = float(prob_vec[best_id])
            best_char = self.i2c.get(best_id, "<UNK>")

            if best_char in ("<UNK>", "<PAD>", "<MASK>"):
                applied = False
            elif best_char == chars[i]:
                applied = False
            elif best_p < self.apply_threshold:
                applied = False
            elif best_char.lower() != best_char or chars[i].lower() != chars[i]:
                if best_char.lower() == chars[i].lower():
                    applied = False
                else:
                    test_chars = chars.copy()
                    test_chars[i] = best_char
                    test_word = "".join(test_chars)
                    if self.lexicon and test_word in self.lexicon:
                        applied = True
                    elif self.lexicon:
                        applied = False
                    else:
                        applied = True
            else:
                test_chars = chars.copy()
                test_chars[i] = best_char
                test_word = "".join(test_chars)
                if self.lexicon and test_word in self.lexicon:
                    applied = True
                elif self.lexicon:
                    applied = False
                else:
                    applied = True

            if applied:
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
        max_pairs_edits: int = 3,
        max_len: int = 32,
        emb_size: int = 192,
        n_layers: int = 8,
        n_heads: int = 6,
        ffn_size: int = 1024,
        dropout: float = 0.1,
        batch_size: int = 256,
        accumulation_steps: int = 2,
        use_amp: bool = True,
        compile_model: bool = False,
        epochs: int = 50,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        grad_clip: float = 1.0,
        min_len: int = 3,
        mask_prob: float = 0.3,
        span_min: int = 1,
        span_max: int = 3,
        spans_min: int = 1,
        spans_max: int = 2,
        pairs_ratio: float = 0.8,
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
        max_pairs_edits : int, optional
            Maximum number of character edits in pairs to include. Default is 3.
        max_len : int, optional
            Maximum sequence length. Default is 32.
        emb_size : int, optional
            Embedding size. Default is 192.
        n_layers : int, optional
            Number of transformer layers. Default is 8.
        n_heads : int, optional
            Number of attention heads. Default is 6.
        ffn_size : int, optional
            Feed-forward network size. Default is 1024.
        dropout : float, optional
            Dropout rate. Default is 0.1.
        batch_size : int, optional
            Batch size. Default is 256.
        accumulation_steps : int, optional
            Gradient accumulation steps. Default is 2.
        use_amp : bool, optional
            Use automatic mixed precision (AMP). Default is True.
        compile_model : bool, optional
            Use torch.compile for faster training. Default is False.
        epochs : int, optional
            Number of epochs. Default is 50.
        lr : float, optional
            Learning rate. Default is 1e-3.
        weight_decay : float, optional
            Weight decay. Default is 0.01.
        grad_clip : float, optional
            Gradient clipping. Default is 1.0.
        min_len : int, optional
            Minimum word length. Default is 3.
        mask_prob : float, optional
            Probability of using span masking. Default is 0.3.
        span_min : int, optional
            Minimum span length for masking. Default is 1.
        span_max : int, optional
            Maximum span length for masking. Default is 3.
        spans_min : int, optional
            Minimum number of spans. Default is 1.
        spans_max : int, optional
            Maximum number of spans. Default is 2.
        pairs_ratio : float, optional
            Ratio of OCR pairs in mixed dataset (0.8 = 80% pairs, 20% ngrams). Default is 0.8.
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
            Path to the final checkpoint. After training, CharLM also attempts
            to export the final checkpoint to ONNX in ``exp_dir``.
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
            "max_pairs_edits": max_pairs_edits,
            "max_len": max_len,
            "emb_size": emb_size,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "ffn_size": ffn_size,
            "dropout": dropout,
            "batch_size": batch_size,
            "accumulation_steps": accumulation_steps,
            "use_amp": use_amp,
            "compile_model": compile_model,
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
            "pairs_ratio": pairs_ratio,
            "eval_ratio": eval_ratio,
            "seed": seed,
            "checkpoint": checkpoint,
            **extra_config,
        }

        run_training(config)

        final_checkpoint_path = os.path.join(
            exp_dir, "checkpoints", f"charlm_epoch_{epochs}.pt"
        )

        try:
            print("Attempting to export final model to ONNX...")
            onnx_path = os.path.join(exp_dir, f"charlm_epoch_{epochs}.onnx")

            CharLM.export(
                weights_path=final_checkpoint_path,
                vocab_path=os.path.join(exp_dir, "vocab.json"),
                output_path=onnx_path,
                max_len=config["max_len"],
                emb_size=config["emb_size"],
                n_layers=config["n_layers"],
                n_heads=config["n_heads"],
                ffn_size=config["ffn_size"],
                opset_version=14,
                simplify=True,
            )
            print(f"ONNX model exported successfully: {onnx_path}")
        except Exception as e:
            print(f"Failed to export ONNX model: {e}")

        return final_checkpoint_path

    @staticmethod
    def export(
        weights_path: Union[str, Path],
        vocab_path: Union[str, Path],
        output_path: Union[str, Path],
        max_len: int = 32,
        emb_size: int = 192,
        n_layers: int = 8,
        n_heads: int = 6,
        ffn_size: int = 1024,
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
            Number of transformer layers. Default is 8.
        n_heads : int, optional
            Number of attention heads. Default is 6.
        ffn_size : int, optional
            Feed-forward network size. Default is 1024.
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

        print(f"Loading vocab from {vocab_path}...")
        with open(vocab_path, "r", encoding="utf-8") as f:
            chars = json.load(f)
        vocab_size = len(chars)
        c2i = {c: i for i, c in enumerate(chars)}
        pad_idx = c2i.get("<PAD>", 0)

        print(f"Vocab size: {vocab_size}")

        print(f"Loading checkpoint from {weights_path}...")
        checkpoint = torch.load(str(weights_path), map_location="cpu")
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        model = CharTransformerMLM(
            vocab_size=vocab_size,
            emb_size=emb_size,
            max_len=max_len,
            n_layers=n_layers,
            n_heads=n_heads,
            ffn_size=ffn_size,
            dropout=0.0,
            pad_idx=pad_idx,
        )
        model.load_state_dict(state_dict)
        model.eval()

        print(f"\n=== CharLM ONNX Export ===")
        print(f"Max length: {max_len}")
        print(f"Embedding size: {emb_size}")
        print(f"Layers: {n_layers}")
        print(f"Heads: {n_heads}")

        dummy_input = torch.full((1, max_len), c2i.get("<MASK>", 1), dtype=torch.long)

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
            dynamo=False,
        )

        print(f"[OK] ONNX model saved to: {output_path}")

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
