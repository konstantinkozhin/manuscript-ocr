"""Reliability tests for TRBA transform layer."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("albumentations")

from manuscript.recognizers._trba.data.transforms import (
    DEFAULT_AUG_PARAMS,
    BBoxJitter,
    DownscaleUpscale,
    GaussianNoiseCustom,
    RandomErasing,
    RandomZoomIn,
    RandomZoomOut,
    ResizeAndPadA,
    SaltPepperNoise,
    StrokeWidthPerturbation,
    TextWarp,
    get_train_transform,
    get_val_transform,
    list_augmentations,
    preview_augmentation,
)


def _rgb_image(height=20, width=40, value=120):
    return np.full((height, width, 3), value, dtype=np.uint8)


class TestTRBAResizeAndPad:
    def test_resize_and_pad_handles_grayscale_and_alignment(self):
        img = np.full((10, 20), 80, dtype=np.uint8)

        out = ResizeAndPadA(
            img_h=32,
            img_w=64,
            align_h="right",
            align_v="bottom",
        ).apply(img)

        assert out.shape == (32, 64, 3)
        assert out.dtype == np.uint8
        assert np.any(out[-10:, -20:, :] == 80)

    def test_resize_and_pad_handles_rgba(self):
        img = np.dstack(
            [
                np.full((12, 12), 10, dtype=np.uint8),
                np.full((12, 12), 20, dtype=np.uint8),
                np.full((12, 12), 30, dtype=np.uint8),
                np.full((12, 12), 255, dtype=np.uint8),
            ]
        )

        out = ResizeAndPadA(img_h=24, img_w=24).apply(img)

        assert out.shape == (24, 24, 3)
        assert out.dtype == np.uint8


class TestTRBACustomTransforms:
    @pytest.mark.parametrize(
        "factory",
        [
            lambda: RandomZoomOut(scale_min=0.6, scale_max=0.6, p=1.0),
            lambda: RandomZoomIn(scale_min=1.2, scale_max=1.2, p=1.0),
            lambda: BBoxJitter(offset=0.1, p=1.0),
            lambda: TextWarp(strength=0.02, p=1.0),
            lambda: StrokeWidthPerturbation(kernel_min=1, kernel_max=1, p=1.0),
            lambda: GaussianNoiseCustom(std_min=0.01, std_max=0.01, p=1.0),
            lambda: SaltPepperNoise(amount_min=0.01, amount_max=0.01, p=1.0),
            lambda: DownscaleUpscale(scale_min=0.7, scale_max=0.7, p=1.0),
            lambda: RandomErasing(num_holes=1, size_min=0.1, size_max=0.1, p=1.0),
        ],
    )
    def test_custom_transforms_preserve_shape_dtype_and_finiteness(self, factory):
        img = _rgb_image()

        out = factory().apply(img)

        assert out.shape == img.shape
        assert out.dtype == np.uint8
        assert np.all(np.isfinite(out))


class TestTRBAComposeTransforms:
    def test_get_train_transform_returns_tensor_with_expected_shape(self):
        params = dict(DEFAULT_AUG_PARAMS)
        for key in list(params):
            if key.startswith("p_") or key == "invert_p":
                params[key] = 0.0

        transform = get_train_transform(params=params, img_h=32, img_w=64)
        out = transform(image=_rgb_image(height=18, width=30))["image"]

        assert isinstance(out, torch.Tensor)
        assert tuple(out.shape) == (3, 32, 64)
        assert torch.isfinite(out).all()

    def test_get_val_transform_returns_tensor_with_expected_shape(self):
        transform = get_val_transform(img_h=32, img_w=64)
        out = transform(image=_rgb_image(height=18, width=30))["image"]

        assert isinstance(out, torch.Tensor)
        assert tuple(out.shape) == (3, 32, 64)
        assert torch.isfinite(out).all()


class TestTRBAPreviewAugmentation:
    def test_list_augmentations_contains_expected_names(self):
        names = list_augmentations()

        assert "original" in names
        assert "text_mosaic" in names
        assert "gaussian_noise" in names
        assert len(names) == len(set(names))

    def test_preview_augmentation_original_returns_copy(self):
        img = _rgb_image()

        out = preview_augmentation(img, "original")

        assert out.shape == img.shape
        assert out.dtype == np.uint8
        assert np.array_equal(out, img)
        assert out is not img

    def test_preview_augmentation_text_mosaic_increases_width(self):
        img = _rgb_image(height=16, width=20)

        out = preview_augmentation(img, "text_mosaic")

        assert out.shape[0] == img.shape[0]
        assert out.shape[1] > img.shape[1]
        assert out.dtype == np.uint8

    def test_preview_augmentation_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown augmentation name"):
            preview_augmentation(_rgb_image(), "missing_aug")
