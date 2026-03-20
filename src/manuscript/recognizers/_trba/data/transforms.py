import os
from collections import defaultdict

import cv2
import numpy as np

# Optional imports for training (not needed for inference)
try:
    import torch
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    A = None
    ToTensorV2 = None
    _TORCH_AVAILABLE = False

# ──────────────────────────────────────────────────────────────────
# Augmentation parameter defaults
# ──────────────────────────────────────────────────────────────────
DEFAULT_AUG_PARAMS = {
    # --- Geometric ---
    "p_ShiftScaleRotate": 0.3,
    "shift_limit": 0.03,
    "scale_limit": 0.08,
    "rotate_limit": 3,

    "p_Perspective": 0.25,
    "perspective_scale_min": 0.02,
    "perspective_scale_max": 0.05,

    "p_TextWarp": 0.2,
    "text_warp_strength": 0.03,

    "p_ZoomOut": 0.25,
    "zoom_out_scale_min": 0.65,
    "zoom_out_scale_max": 1.0,

    "p_ZoomIn": 0.2,
    "zoom_in_scale_min": 1.0,
    "zoom_in_scale_max": 1.25,

    "p_BBoxJitter": 0.25,
    "bbox_jitter_offset": 0.12,

    # --- Morphology ---
    "p_StrokeWidth": 0.1,
    "stroke_kernel_min": 1,
    "stroke_kernel_max": 2,

    # --- Photometric ---
    "p_BrightnessContrast": 0.3,
    "brightness_limit": 0.2,
    "contrast_limit": 0.2,

    "p_Gamma": 0.2,
    "gamma_min": 70,   # albumentations gamma range is int [1,200], 100=neutral
    "gamma_max": 150,

    "invert_p": 0.08,

    # --- Blur ---
    "p_GaussianBlur": 0.2,
    "blur_kernel_min": 3,
    "blur_kernel_max": 5,

    "p_MotionBlur": 0.15,
    "motion_blur_kernel_min": 3,
    "motion_blur_kernel_max": 7,

    # --- Noise ---
    "p_GaussNoise": 0.2,
    "gauss_noise_std_min": 0.01,
    "gauss_noise_std_max": 0.03,

    "p_SaltPepper": 0.1,
    "salt_pepper_amount_min": 0.001,
    "salt_pepper_amount_max": 0.005,

    # --- Compression / resolution ---
    "p_JPEG": 0.15,
    "jpeg_quality_min": 60,
    "jpeg_quality_max": 95,

    "p_Downscale": 0.2,
    "downscale_min": 0.5,
    "downscale_max": 0.9,

    # --- Occlusion ---
    "p_RandomErasing": 0.1,
    "erasing_holes": 2,
    "erasing_size_min": 0.05,
    "erasing_size_max": 0.15,
}


def build_file_index(roots, exts={".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}):
    """Builds a file index for fast lookup."""
    if isinstance(roots, str):
        roots = [roots]
    index = defaultdict(list)
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                ext = os.path.splitext(fn)[1].lower()
                if exts and ext not in exts:
                    continue
                index[fn.lower()].append(os.path.join(dirpath, fn))
    return index


def load_charset(charset_path: str):
    """
    Loads the character vocabulary from a file of the format:
        <PAD>
        <SOS>
        <EOS>
        <BLANK>
        a
        b
        ...
    Returns (itos, stoi).
    """
    itos = []
    with open(charset_path, "r", encoding="utf-8") as f:
        for line in f:
            tok = line.rstrip("\n")
            if tok == "":
                continue
            itos.append(tok)
    stoi = {s: i for i, s in enumerate(itos)}
    return itos, stoi


# Training-only classes and functions (require albumentations and torch)
if _TORCH_AVAILABLE:

    class ResizeAndPadA(A.ImageOnlyTransform):
        """Custom transform for resizing and padding."""

        def __init__(
            self,
            img_h=32,
            img_w=256,
            align_h="left",
            align_v="center",
            always_apply=True,
            p=1.0,
        ):
            super().__init__(always_apply, p)
            self.img_h = int(img_h)
            self.img_w = int(img_w)
            self.align_h = align_h
            self.align_v = align_v

        def _interp(self, src_h, src_w, dst_h, dst_w):
            if dst_h < src_h or dst_w < src_w:
                return cv2.INTER_AREA
            return cv2.INTER_LINEAR

        def apply(self, img, **params):
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

            h, w = img.shape[:2]

            scale = min(self.img_h / max(h, 1), self.img_w / max(w, 1))
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))

            interp = self._interp(h, w, new_h, new_w)
            img_resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

            canvas = np.full((self.img_h, self.img_w, 3), 255, dtype=img.dtype)

            if self.align_h == "left":
                x0 = 0
            elif self.align_h == "right":
                x0 = self.img_w - new_w
            else:
                x0 = (self.img_w - new_w) // 2

            if self.align_v == "top":
                y0 = 0
            elif self.align_v == "bottom":
                y0 = self.img_h - new_h
            else:
                y0 = (self.img_h - new_h) // 2

            x0 = max(0, min(x0, self.img_w - new_w))
            y0 = max(0, min(y0, self.img_h - new_h))

            canvas[y0 : y0 + new_h, x0 : x0 + new_w] = img_resized
            return canvas

    # ── Detector-noise simulation ────────────────────────────────────────────

    class RandomZoomOut(A.ImageOnlyTransform):
        """Shrink text inside the crop box — simulates detector padding it with whitespace."""

        def __init__(self, scale_min=0.65, scale_max=1.0, p=0.25):
            super().__init__(p=p)
            self.scale_min = scale_min
            self.scale_max = scale_max

        def apply(self, img, **params):
            h, w = img.shape[:2]
            scale = np.random.uniform(self.scale_min, self.scale_max)
            if scale >= 1.0:
                return img
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            canvas = np.full((h, w, 3), 255, dtype=np.uint8)
            x0 = (w - new_w) // 2
            y0 = (h - new_h) // 2
            canvas[y0: y0 + new_h, x0: x0 + new_w] = small
            return canvas

        def get_transform_init_args_names(self):
            return ("scale_min", "scale_max")

    class RandomZoomIn(A.ImageOnlyTransform):
        """Zoom into text — some characters may be clipped; simulates tight detector crop."""

        def __init__(self, scale_min=1.0, scale_max=1.25, p=0.2):
            super().__init__(p=p)
            self.scale_min = scale_min
            self.scale_max = scale_max

        def apply(self, img, **params):
            h, w = img.shape[:2]
            scale = np.random.uniform(self.scale_min, self.scale_max)
            if scale <= 1.0:
                return img
            new_w = int(w * scale)
            new_h = int(h * scale)
            big = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            x0 = (new_w - w) // 2
            y0 = (new_h - h) // 2
            return big[y0: y0 + h, x0: x0 + w]

        def get_transform_init_args_names(self):
            return ("scale_min", "scale_max")

    class BBoxJitter(A.ImageOnlyTransform):
        """Random pixel-offset crop/pad — simulates bbox misalignment from detector."""

        def __init__(self, offset=0.12, p=0.25):
            super().__init__(p=p)
            self.offset = offset

        def apply(self, img, **params):
            h, w = img.shape[:2]
            dx = int(np.random.uniform(-self.offset, self.offset) * w)
            dy = int(np.random.uniform(-self.offset, self.offset) * h)
            # Shift by dx/dy using border padding
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            return cv2.warpAffine(
                img, M, (w, h),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255),
            )

        def get_transform_init_args_names(self):
            return ("offset",)

    # ── OCR-specific geometric ────────────────────────────────────────────────

    class TextWarp(A.ImageOnlyTransform):
        """Thin-plate-spline-like sinusoidal warp — simulates curved pages / banners."""

        def __init__(self, strength=0.03, p=0.2):
            super().__init__(p=p)
            self.strength = strength

        def apply(self, img, **params):
            h, w = img.shape[:2]
            amp_x = self.strength * w * np.random.uniform(0.5, 1.5)
            amp_y = self.strength * h * np.random.uniform(0.5, 1.5)
            freq = np.random.uniform(0.5, 1.5)

            x, y = np.meshgrid(np.arange(w), np.arange(h))
            map_x = (x + amp_x * np.sin(2 * np.pi * y / max(h, 1) * freq)).astype(np.float32)
            map_y = (y + amp_y * np.sin(2 * np.pi * x / max(w, 1) * freq)).astype(np.float32)
            map_x = np.clip(map_x, 0, w - 1)
            map_y = np.clip(map_y, 0, h - 1)
            return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(255, 255, 255))

        def get_transform_init_args_names(self):
            return ("strength",)

    # ── Morphology ───────────────────────────────────────────────────────────

    class StrokeWidthPerturbation(A.ImageOnlyTransform):
        """Random dilation or erosion — varies stroke width."""

        def __init__(self, kernel_min=1, kernel_max=2, p=0.1):
            super().__init__(p=p)
            self.kernel_min = kernel_min
            self.kernel_max = kernel_max

        def apply(self, img, **params):
            k = np.random.randint(self.kernel_min, self.kernel_max + 1)
            kernel = np.ones((k, k), dtype=np.uint8)
            op = cv2.dilate if np.random.rand() < 0.5 else cv2.erode
            return op(img, kernel, iterations=1)

        def get_transform_init_args_names(self):
            return ("kernel_min", "kernel_max")

    # ── Noise ────────────────────────────────────────────────────────────────

    class GaussianNoiseCustom(A.ImageOnlyTransform):
        """Gaussian noise with std sampled from [std_min, std_max] * 255."""

        def __init__(self, std_min=0.01, std_max=0.03, p=0.2):
            super().__init__(p=p)
            self.std_min = std_min
            self.std_max = std_max

        def apply(self, img, **params):
            std = np.random.uniform(self.std_min, self.std_max) * 255.0
            noise = np.random.normal(0.0, std, img.shape).astype(np.float32)
            out = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            return out

        def get_transform_init_args_names(self):
            return ("std_min", "std_max")

    class SaltPepperNoise(A.ImageOnlyTransform):
        """Salt-and-pepper noise."""

        def __init__(self, amount_min=0.001, amount_max=0.005, p=0.1):
            super().__init__(p=p)
            self.amount_min = amount_min
            self.amount_max = amount_max

        def apply(self, img, **params):
            out = img.copy()
            amount = np.random.uniform(self.amount_min, self.amount_max)
            rnd = np.random.rand(*img.shape[:2])
            out[rnd < amount] = 0
            out[rnd > 1 - amount] = 255
            return out

        def get_transform_init_args_names(self):
            return ("amount_min", "amount_max")

    # ── Downscale (custom: scale down then up) ────────────────────────────────

    class DownscaleUpscale(A.ImageOnlyTransform):
        """Downscale then upscale to simulate low-resolution text."""

        def __init__(self, scale_min=0.5, scale_max=0.9, p=0.2):
            super().__init__(p=p)
            self.scale_min = scale_min
            self.scale_max = scale_max

        def apply(self, img, **params):
            h, w = img.shape[:2]
            scale = np.random.uniform(self.scale_min, self.scale_max)
            nw, nh = max(2, int(w * scale)), max(2, int(h * scale))
            small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
            return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

        def get_transform_init_args_names(self):
            return ("scale_min", "scale_max")

    # ── Occlusion ────────────────────────────────────────────────────────────

    class RandomErasing(A.ImageOnlyTransform):
        """Fill random rectangles with mean color — simulates partial occlusion."""

        def __init__(self, num_holes=2, size_min=0.05, size_max=0.15, p=0.1):
            super().__init__(p=p)
            self.num_holes = num_holes
            self.size_min = size_min
            self.size_max = size_max

        def apply(self, img, **params):
            out = img.copy()
            h, w = out.shape[:2]
            fill = out.mean(axis=(0, 1)).astype(np.uint8)
            for _ in range(self.num_holes):
                hw = int(np.random.uniform(self.size_min, self.size_max) * w)
                hh = int(np.random.uniform(self.size_min, self.size_max) * h)
                x0 = np.random.randint(0, max(1, w - hw + 1))
                y0 = np.random.randint(0, max(1, h - hh + 1))
                out[y0: y0 + hh, x0: x0 + hw] = fill
            return out

        def get_transform_init_args_names(self):
            return ("num_holes", "size_min", "size_max")

else:
    # Stubs for when torch is not available
    ResizeAndPadA = None
    RandomZoomOut = None
    RandomZoomIn = None
    BBoxJitter = None
    TextWarp = None
    StrokeWidthPerturbation = None
    GaussianNoiseCustom = None
    SaltPepperNoise = None
    DownscaleUpscale = None
    RandomErasing = None


def pack_attention_targets(texts, stoi, max_len, drop_blank=True):
    """Packing text targets for attention model (requires torch)."""
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "pack_attention_targets requires PyTorch. Install with: pip install manuscript-ocr[dev]"
        )

    PAD = stoi["<PAD>"]
    SOS = stoi["<SOS>"]
    EOS = stoi["<EOS>"]
    BLANK = stoi.get("<BLANK>", None)

    B = len(texts)
    T = max_len + 1

    text_in = torch.full((B, T), PAD, dtype=torch.long)
    text_in[:, 0] = SOS

    target_y = torch.full((B, T), PAD, dtype=torch.long)
    lengths = torch.zeros(B, dtype=torch.long)

    for i, s in enumerate(texts):
        ids = []
        for ch in s:
            if ch not in stoi:
                continue
            idx = stoi[ch]
            if drop_blank and BLANK is not None and idx == BLANK:
                continue
            ids.append(idx)

        L = min(len(ids), max_len)
        if L > 0:
            text_in[i, 1 : 1 + L] = torch.tensor(ids[:L], dtype=torch.long)
            target_y[i, :L] = torch.tensor(ids[:L], dtype=torch.long)

        target_y[i, L] = EOS
        lengths[i] = L + 1

    return text_in, target_y, lengths


def get_train_transform(params, img_h, img_w):
    """Get training transforms (requires albumentations and torch).

    All augmentation probabilities and strengths are configurable via ``params``
    (a dict or Config object).  Defaults are in ``DEFAULT_AUG_PARAMS``.

    Pipeline order (recommended by OCR literature):
    1. ResizeAndPadA            — always
    2. RandomZoomOut            — detector noise
    3. RandomZoomIn             — detector noise
    4. BBoxJitter               — detector noise
    5. Affine (shift/scale/rot) — geometric
    6. Perspective              — geometric
    7. TextWarp                 — OCR-specific warp
    8. StrokeWidthPerturbation  — morphology
    9. RandomBrightnessContrast — photometric
    10. RandomGamma             — photometric
    11. InvertImg               — photometric
    12. GaussianBlur            — blur
    13. MotionBlur              — blur
    14. GaussianNoiseCustom     — noise
    15. SaltPepperNoise         — noise
    16. ImageCompression (JPEG) — compression
    17. DownscaleUpscale        — resolution
    18. RandomErasing           — occlusion
    19. Normalize               — always
    20. ToTensorV2              — always
    """
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "get_train_transform requires PyTorch. Install with: pip install manuscript-ocr[dev]"
        )

    def _p(key):
        return float(params.get(key, DEFAULT_AUG_PARAMS[key]))

    def _v(key):
        return params.get(key, DEFAULT_AUG_PARAMS[key])

    return A.Compose(
        [
            # 1. Resize + pad (always)
            ResizeAndPadA(img_h=img_h, img_w=img_w),

            # 2-4. Detector-noise simulation
            RandomZoomOut(
                scale_min=float(_v("zoom_out_scale_min")),
                scale_max=float(_v("zoom_out_scale_max")),
                p=_p("p_ZoomOut"),
            ),
            RandomZoomIn(
                scale_min=float(_v("zoom_in_scale_min")),
                scale_max=float(_v("zoom_in_scale_max")),
                p=_p("p_ZoomIn"),
            ),
            BBoxJitter(
                offset=float(_v("bbox_jitter_offset")),
                p=_p("p_BBoxJitter"),
            ),

            # 5. Geometric: Affine (shift + scale + rotate)
            A.Affine(
                translate_percent={
                    "x": (-round(float(_v("shift_limit")), 4), round(float(_v("shift_limit")), 4)),
                    "y": (-round(float(_v("shift_limit")), 4), round(float(_v("shift_limit")), 4)),
                },
                scale=(
                    round(1.0 - float(_v("scale_limit")), 4),
                    round(1.0 + float(_v("scale_limit")), 4),
                ),
                rotate=(-int(_v("rotate_limit")), int(_v("rotate_limit"))),
                border_mode=0,
                fill=(255, 255, 255),
                p=_p("p_ShiftScaleRotate"),
            ),

            # 6. Perspective distortion
            A.Perspective(
                scale=(
                    float(_v("perspective_scale_min")),
                    float(_v("perspective_scale_max")),
                ),
                fill=255,
                p=_p("p_Perspective"),
            ),

            # 7. OCR-specific text warp
            TextWarp(
                strength=float(_v("text_warp_strength")),
                p=_p("p_TextWarp"),
            ),

            # 8. Stroke width (dilation/erosion)
            StrokeWidthPerturbation(
                kernel_min=int(_v("stroke_kernel_min")),
                kernel_max=int(_v("stroke_kernel_max")),
                p=_p("p_StrokeWidth"),
            ),

            # 9-11. Photometric
            A.RandomBrightnessContrast(
                brightness_limit=round(float(_v("brightness_limit")), 4),
                contrast_limit=round(float(_v("contrast_limit")), 4),
                p=_p("p_BrightnessContrast"),
            ),
            A.RandomGamma(
                gamma_limit=(int(_v("gamma_min")), int(_v("gamma_max"))),
                p=_p("p_Gamma"),
            ),
            A.InvertImg(p=_p("invert_p")),

            # 12-13. Blur
            A.GaussianBlur(
                blur_limit=(int(_v("blur_kernel_min")), int(_v("blur_kernel_max"))),
                p=_p("p_GaussianBlur"),
            ),
            A.MotionBlur(
                blur_limit=(
                    int(_v("motion_blur_kernel_min")),
                    int(_v("motion_blur_kernel_max")),
                ),
                p=_p("p_MotionBlur"),
            ),

            # 14-15. Noise
            GaussianNoiseCustom(
                std_min=float(_v("gauss_noise_std_min")),
                std_max=float(_v("gauss_noise_std_max")),
                p=_p("p_GaussNoise"),
            ),
            SaltPepperNoise(
                amount_min=float(_v("salt_pepper_amount_min")),
                amount_max=float(_v("salt_pepper_amount_max")),
                p=_p("p_SaltPepper"),
            ),

            # 16-17. Compression / resolution
            A.ImageCompression(
                quality_range=(int(_v("jpeg_quality_min")), int(_v("jpeg_quality_max"))),
                p=_p("p_JPEG"),
            ),
            DownscaleUpscale(
                scale_min=float(_v("downscale_min")),
                scale_max=float(_v("downscale_max")),
                p=_p("p_Downscale"),
            ),

            # 18. Occlusion
            RandomErasing(
                num_holes=int(_v("erasing_holes")),
                size_min=float(_v("erasing_size_min")),
                size_max=float(_v("erasing_size_max")),
                p=_p("p_RandomErasing"),
            ),

            # 19-20. Normalization (always)
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ]
    )


def get_val_transform(img_h, img_w):
    """Get validation transforms (requires albumentations and torch)."""
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "get_val_transform requires PyTorch. Install with: pip install manuscript-ocr[dev]"
        )
    return A.Compose(
        [
            ResizeAndPadA(img_h=img_h, img_w=img_w),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ]
    )


def list_augmentations():
    """Return the ordered list of all supported training augmentation names."""
    return [
        "original",
        "zoom_out",
        "zoom_in",
        "bbox_jitter",
        "shift_scale_rotate",
        "perspective",
        "text_warp",
        "stroke_width",
        "brightness_contrast",
        "gamma",
        "invert",
        "gaussian_blur",
        "motion_blur",
        "gaussian_noise",
        "salt_pepper",
        "jpeg",
        "downscale",
        "random_erasing",
        "text_mosaic",
    ]


# ──────────────────────────────────────────────────────────────────
# Text Mosaic: horizontal word concatenation
# ──────────────────────────────────────────────────────────────────

def make_text_mosaic(images, gap_ratio=None):
    """Concatenate 2–3 word-crop images horizontally with a white gap.

    Parameters
    ----------
    images : list of np.ndarray
        List of RGB uint8 word images (can have different sizes).
        Typically 2 or 3 items.
    gap_ratio : float or None
        Gap width as a fraction of the image height.
        If None (default), a random value in [0.03, 0.05] is sampled each call,
        simulating natural inter-word spacing variation.

    Returns
    -------
    np.ndarray
        Single RGB uint8 image with all words laid out left-to-right.
    """
    if not images:
        raise ValueError("images list must not be empty")

    # Resize all to the same height (max height across all images)
    max_h = max(img.shape[0] for img in images)

    resized = []
    for img in images:
        h, w = img.shape[:2]
        if h != max_h:
            scale = max_h / h
            new_w = max(1, int(w * scale))
            img = cv2.resize(img, (new_w, max_h), interpolation=cv2.INTER_LINEAR)
        resized.append(img)

    # Рандомный пробел между словами: от 3% до 5% высоты (имитирует реальный межсловный интервал)
    if gap_ratio is None:
        gap_ratio = np.random.uniform(0.03, 0.05)
    gap_w = max(1, int(max_h * gap_ratio))
    gap = np.full((max_h, gap_w, 3), 255, dtype=np.uint8)

    parts = []
    for i, img in enumerate(resized):
        parts.append(img)
        if i < len(resized) - 1:
            parts.append(gap)

    return np.concatenate(parts, axis=1)


def preview_augmentation(img: np.ndarray, name: str, params: dict = None) -> np.ndarray:
    """Apply a single named augmentation to ``img`` and return the result.

    Parameters
    ----------
    img : np.ndarray
        Input RGB uint8 image.
    name : str
        Augmentation name from ``list_augmentations()``.
    params : dict, optional
        Augmentation parameters (same keys as ``DEFAULT_AUG_PARAMS``).

    Returns
    -------
    np.ndarray
        Augmented RGB uint8 image (no normalization / tensor conversion).
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("preview_augmentation requires albumentations.")

    p = dict(DEFAULT_AUG_PARAMS)
    if params:
        p.update(params)

    name = name.lower().strip()

    def _apply(transform):
        out = transform(image=img)["image"]
        if out.dtype != np.uint8:
            out = np.clip(out, 0, 255).astype(np.uint8)
        return out

    if name == "original":
        return img.copy()
    elif name == "zoom_out":
        return _apply(RandomZoomOut(scale_min=p["zoom_out_scale_min"], scale_max=p["zoom_out_scale_max"], p=1.0))
    elif name == "zoom_in":
        return _apply(RandomZoomIn(scale_min=p["zoom_in_scale_min"], scale_max=p["zoom_in_scale_max"], p=1.0))
    elif name == "bbox_jitter":
        return _apply(BBoxJitter(offset=p["bbox_jitter_offset"], p=1.0))
    elif name == "shift_scale_rotate":
        sl = p["shift_limit"]
        return _apply(A.Affine(
            translate_percent={"x": (-sl, sl), "y": (-sl, sl)},
            scale=(1.0 - p["scale_limit"], 1.0 + p["scale_limit"]),
            rotate=(-p["rotate_limit"], p["rotate_limit"]),
            border_mode=0, fill=(255, 255, 255), p=1.0))
    elif name == "perspective":
        return _apply(A.Perspective(
            scale=(p["perspective_scale_min"], p["perspective_scale_max"]),
            fill=255, p=1.0))
    elif name == "text_warp":
        return _apply(TextWarp(strength=p["text_warp_strength"], p=1.0))
    elif name == "stroke_width":
        return _apply(StrokeWidthPerturbation(
            kernel_min=p["stroke_kernel_min"], kernel_max=p["stroke_kernel_max"], p=1.0))
    elif name == "brightness_contrast":
        return _apply(A.RandomBrightnessContrast(
            brightness_limit=p["brightness_limit"], contrast_limit=p["contrast_limit"], p=1.0))
    elif name == "gamma":
        return _apply(A.RandomGamma(gamma_limit=(p["gamma_min"], p["gamma_max"]), p=1.0))
    elif name == "invert":
        return _apply(A.InvertImg(p=1.0))
    elif name == "gaussian_blur":
        return _apply(A.GaussianBlur(blur_limit=(p["blur_kernel_min"], p["blur_kernel_max"]), p=1.0))
    elif name == "motion_blur":
        return _apply(A.MotionBlur(blur_limit=(p["motion_blur_kernel_min"], p["motion_blur_kernel_max"]), p=1.0))
    elif name == "gaussian_noise":
        return _apply(GaussianNoiseCustom(std_min=p["gauss_noise_std_min"], std_max=p["gauss_noise_std_max"], p=1.0))
    elif name == "salt_pepper":
        return _apply(SaltPepperNoise(amount_min=p["salt_pepper_amount_min"], amount_max=p["salt_pepper_amount_max"], p=1.0))
    elif name == "jpeg":
        return _apply(A.ImageCompression(quality_range=(p["jpeg_quality_min"], p["jpeg_quality_max"]), p=1.0))
    elif name == "downscale":
        return _apply(DownscaleUpscale(scale_min=p["downscale_min"], scale_max=p["downscale_max"], p=1.0))
    elif name == "random_erasing":
        return _apply(RandomErasing(
            num_holes=p["erasing_holes"], size_min=p["erasing_size_min"],
            size_max=p["erasing_size_max"], p=1.0))
    elif name == "text_mosaic":
        # Return the image concatenated with itself (demo: single-image mosaic)
        return make_text_mosaic([img, img])
    else:
        raise ValueError(f"Unknown augmentation name: '{name}'. "
                         f"Use list_augmentations() to see available names.")


def decode_tokens(ids, itos, pad_id, eos_id, blank_id=None):
    """Decoding tokens into text."""
    out = []
    for t in ids:
        t = int(t)
        if t == eos_id:
            break
        if t == pad_id or (blank_id is not None and t == blank_id):
            continue
        out.append(itos[t])
    return "".join(out)
