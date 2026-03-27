"""Reliability tests for TRBA data/transforms layer."""

import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")

import manuscript.recognizers._trba.data.dataset as trba_dataset_module
from manuscript.recognizers._trba.data.dataset import (
    MultiDataset,
    OCRDatasetAttn,
    ProportionalBatchSampler,
)
from manuscript.recognizers._trba.data.transforms import (
    build_file_index,
    decode_tokens,
    load_charset,
    make_text_mosaic,
    pack_attention_targets,
)


def _write_image(path: Path, width: int = 24, height: int = 12, value: int = 180) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.full((height, width, 3), value, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _write_csv(path: Path, rows, delimiter=",") -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerows(rows)


class TestTRBATransforms:
    def test_build_file_index_supports_multiple_roots_and_filters_extensions(
        self, tmp_path
    ):
        root_a = tmp_path / "a"
        root_b = tmp_path / "b"
        _write_image(root_a / "Alpha.JPG")
        _write_image(root_b / "beta.png")
        (root_b / "notes.txt").write_text("ignore me", encoding="utf-8")

        index = build_file_index([str(root_a), str(root_b)])

        assert "alpha.jpg" in index
        assert "beta.png" in index
        assert "notes.txt" not in index
        assert index["alpha.jpg"][0].lower().endswith("alpha.jpg")

    def test_load_charset_skips_empty_lines(self, tmp_path):
        charset_path = tmp_path / "charset.txt"
        charset_path.write_text("<PAD>\n<SOS>\n\n<EOS>\na\n", encoding="utf-8")

        itos, stoi = load_charset(str(charset_path))

        assert itos == ["<PAD>", "<SOS>", "<EOS>", "a"]
        assert stoi["<EOS>"] == 2

    def test_pack_attention_targets_sets_eos_and_skips_unknown_chars(self):
        stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<BLANK>": 3, "a": 4, "b": 5}

        text_in, target_y, lengths = pack_attention_targets(
            ["ab", "a?b"],
            stoi=stoi,
            max_len=4,
        )

        assert tuple(text_in.shape) == (2, 5)
        assert tuple(target_y.shape) == (2, 5)
        assert text_in[0, 0].item() == stoi["<SOS>"]
        assert target_y[0, 2].item() == stoi["<EOS>"]
        assert lengths.tolist() == [3, 3]
        assert text_in[1, 1].item() == stoi["a"]
        assert text_in[1, 2].item() == stoi["b"]

    def test_make_text_mosaic_resizes_and_concatenates(self):
        img_a = np.full((10, 20, 3), 100, dtype=np.uint8)
        img_b = np.full((20, 30, 3), 200, dtype=np.uint8)

        mosaic = make_text_mosaic([img_a, img_b], gap_ratio=0.1)

        assert mosaic.shape[0] == 20
        assert mosaic.shape[1] == 40 + 30 + 2
        assert mosaic.dtype == np.uint8

    def test_decode_tokens_stops_at_eos_and_skips_pad_and_blank(self):
        text = decode_tokens(
            [4, 0, 3, 5, 2, 5],
            itos=["<PAD>", "<SOS>", "<EOS>", "<BLANK>", "a", "b"],
            pad_id=0,
            eos_id=2,
            blank_id=3,
        )

        assert text == "ab"


class TestOCRDatasetAttn:
    @pytest.fixture
    def stoi(self):
        return {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "a": 3, "b": 4, " ": 5}

    def test_dataset_reads_tsv_header_and_filters_invalid_rows(self, tmp_path, stoi):
        img_dir = tmp_path / "images"
        _write_image(img_dir / "sample.png")

        csv_path = tmp_path / "dataset.tsv"
        _write_csv(
            csv_path,
            [
                ["image_path", "text"],
                ["sample.png", "ab"],
                ["missing.png", "ab"],
                ["sample.png", ""],
                ["sample.png", "ac"],
            ],
            delimiter="\t",
        )

        dataset = OCRDatasetAttn(
            csv_path=str(csv_path),
            images_dir=str(img_dir),
            stoi=stoi,
            validate_image=False,
        )

        assert len(dataset) == 1
        assert dataset._delimiter == "\t"
        tensor, label = dataset[0]
        assert tuple(tensor.shape) == (3, 12, 24)
        assert label == "ab"

    def test_dataset_resolves_by_basename_and_records_ambiguity(self, tmp_path, stoi):
        root_a = tmp_path / "a"
        root_b = tmp_path / "b"
        _write_image(root_a / "nested" / "dup.jpg", value=80)
        _write_image(root_b / "other" / "dup.jpg", value=160)

        csv_path = tmp_path / "dataset.csv"
        _write_csv(csv_path, [["missing/subdir/dup.jpg", "ab"]])

        dataset = OCRDatasetAttn(
            csv_path=str(csv_path),
            images_dir=[str(root_a), str(root_b)],
            stoi=stoi,
            validate_image=False,
        )

        assert len(dataset) == 1
        assert dataset._reasons["ambiguous"] == 1
        assert dataset.samples[0][0].lower().endswith("dup.jpg")

    def test_dataset_lazy_validation_skips_unreadable_images(
        self, tmp_path, stoi, monkeypatch
    ):
        img_dir = tmp_path / "images"
        valid_path = img_dir / "valid.jpg"
        broken_path = img_dir / "broken.jpg"
        _write_image(valid_path)
        broken_path.parent.mkdir(parents=True, exist_ok=True)
        broken_path.write_bytes(b"not an image")

        csv_path = tmp_path / "dataset.csv"
        _write_csv(csv_path, [["valid.jpg", "ab"], ["broken.jpg", "ab"]])

        dataset = OCRDatasetAttn(
            csv_path=str(csv_path),
            images_dir=str(img_dir),
            stoi=stoi,
            validate_image=True,
        )
        monkeypatch.setattr(trba_dataset_module.random, "choice", lambda seq: seq[0])

        broken_idx = next(
            idx for idx, (path, _) in enumerate(dataset.samples) if path.endswith("broken.jpg")
        )

        tensor, label = dataset[broken_idx]

        assert tuple(tensor.shape) == (3, 12, 24)
        assert label == "ab"
        assert dataset._invalid_mask[broken_idx] is True

    def test_dataset_text_mosaic_combines_labels(self, tmp_path, stoi, monkeypatch):
        img_dir = tmp_path / "images"
        _write_image(img_dir / "one.jpg", width=20, height=12, value=90)
        _write_image(img_dir / "two.jpg", width=20, height=12, value=180)

        csv_path = tmp_path / "dataset.csv"
        _write_csv(csv_path, [["one.jpg", "ab"], ["two.jpg", "ba"]])

        dataset = OCRDatasetAttn(
            csv_path=str(csv_path),
            images_dir=str(img_dir),
            stoi=stoi,
            validate_image=False,
            text_mosaic_prob=1.0,
            text_mosaic_n_words=2,
            max_len=10,
        )

        base_idx = next(
            idx for idx, (path, _) in enumerate(dataset.samples) if path.endswith("one.jpg")
        )
        other_idx = next(
            idx for idx, (path, _) in enumerate(dataset.samples) if path.endswith("two.jpg")
        )

        choices = iter([other_idx])
        monkeypatch.setattr(trba_dataset_module.random, "random", lambda: 0.0)
        monkeypatch.setattr(
            trba_dataset_module.random, "randrange", lambda n: next(choices)
        )

        tensor, label = dataset[base_idx]

        assert label == f"{dataset.samples[base_idx][1]} {dataset.samples[other_idx][1]}"
        assert tensor.shape[2] > 20


class TestTRBADataCombinators:
    def test_proportional_batch_sampler_respects_batch_mix(self):
        datasets = [list(range(8)), list(range(8))]
        sampler = ProportionalBatchSampler(
            datasets=datasets,
            batch_size=4,
            proportions=[0.75, 0.25],
        )

        batch = next(iter(sampler))
        ds0 = sum(1 for ds_idx, _ in batch if ds_idx == 0)
        ds1 = sum(1 for ds_idx, _ in batch if ds_idx == 1)

        assert len(batch) == 4
        assert ds0 == 3
        assert ds1 == 1

    def test_multidataset_delegates_index_to_child_dataset(self):
        datasets = [["a0", "a1"], ["b0"]]
        multi = MultiDataset(datasets)

        assert len(multi) == 3
        assert multi[(0, 1)] == "a1"
        assert multi[(1, 0)] == "b0"
