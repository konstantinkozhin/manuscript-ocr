# TRBA Model Configuration

## Training and inference use different settings

`TRBA.train()` accepts training options as keyword arguments. To keep those
options in `training.json`, load the file explicitly:

```python
import json
from manuscript.recognizers import TRBA

with open("training.json", encoding="utf-8") as stream:
    options = json.load(stream)
summary = TRBA.train(**options)
```

Paths in this file are resolved from the working directory, not from the JSON
file's directory. Install `manuscript-ocr[dev]` for training and export.

For inference, use the matching config and charset produced by training/export,
alongside the ONNX graph. The inference config must include
`recognition_levels`, unless those levels are explicitly supplied to the
constructor. It must describe the same input size and architecture as the model.
Training dataset paths and optimizer settings are not needed for inference.

```python
recognizer = TRBA(
    weights="runs/example/model.onnx",
    config="runs/example/config.json",
    charset="runs/example/charset.txt",
)
```

Registry presets supply their own supporting files. For example,
`TRBA(weights="trba_lite_g2")` does not require explicit config or charset paths.

## Model dimensions

These training arguments have defaults; override them to choose the model's
input size and hidden dimension.

**`img_h`** – Height of the input image for the model in pixels

**`img_w`** – Width of the input image for the model in pixels

**`hidden_size`** – Hidden layer size in the RNN part of the model

## Decoder and input rectification (0.1.13)

Pass these options to `TRBA.train()` or include them in `training.json`:

| Option | Default | Meaning |
| --- | --- | --- |
| `cnn_backbone` | `"seresnet31"` | Use `"seresnetlite31v2"` for SEResNet31LiteV2. |
| `decoder_type` | `"attention"` | Choose `"attention"` or `"parseq"`. |
| `decoder_layers` | `2` | PARSeq decoder layer count. |
| `decoder_heads` | `8` | PARSeq head count; must divide `hidden_size`. |
| `decoder_ffn` | `1024` | PARSeq feed-forward dimension. |
| `decoder_dropout` | `0.1` | PARSeq decoder dropout. |
| `parseq_permutations` | `6` | Training permutation count; even and at least 2. |
| `parseq_refine_iters` | `1` | Inference refinement iterations; nonnegative. |
| `transformation` | `"none"` | Use `"tps"` for input rectification. |
| `tps_num_fiducial` | `20` | TPS control-point count; even and at least 4. |

Training saves these settings for export. Use the matching saved config and
charset with the exported weights. See the
[recognizer guide](../../../../../docs/sphinx/api/recognizers.rst)
for a complete PARSeq/TPS training and export example.

## Character set for training

**`charset_path`** – Optional path to the character set file. `TRBA.train()` uses the packaged charset when this argument is omitted or `None`.

## Additional parameters

**`max_len`** – Maximum length of the recognized character sequence

**`encoding`** – Text data encoding

## Training parameters

**`train_csvs`** – Paths to CSV or TSV files. The first two columns are image path and transcription; use `filename,text` as a CSV header, or omit the header. TSV uses tabs; other delimiters are detected automatically.

**`train_roots`** – Root folders with images for training

**`val_csvs`** – Paths to validation CSV/TSV files. If omitted, samples are held out from each training dataset using `val_size`. A `null` entry uses that fallback for the corresponding dataset.

**`val_roots`** – Root folders with validation images; supply together with `val_csvs`, with matching list lengths.

**`batch_size`** – Batch size for training

**`epochs`** – Maximum number of training epochs

**`lr`** – Learning rate

**`optimizer`** – Optimization algorithm ("Adam", "SGD", "AdamW")

**`scheduler`** – Learning rate scheduler: "OneCycleLR" (default), "ReduceLROnPlateau", "CosineAnnealingLR", or "None" to disable scheduling

**`weight_decay`** – L2 regularization coefficient

**`momentum`** – Momentum coefficient for the SGD optimizer

**`shift_limit`** – Maximum image shift during augmentation

**`scale_limit`** – Maximum image scaling during augmentation

**`rotate_limit`** – Maximum rotation in degrees for augmentation

**`p_ShiftScaleRotate`** – Probability of applying geometric transformations

**`brightness_limit`** – Brightness variation for augmentation

**`contrast_limit`** – Contrast variation for augmentation

**`p_BrightnessContrast`** – Probability of applying color transformations

**`invert_p`** – Probability of color inversion

**`train_proportions`** – Sampling proportions for multiple datasets, normalized to sum to 1. If omitted, datasets are concatenated and shuffled; their contributions follow their sizes.

**`val_size`** – Number of samples held out from each training dataset without a separate validation set (default 3000). Each such dataset must contain more valid samples than this value. Separate validation sets are used in full.

**`num_workers`** – Number of worker processes for data loading

**`resume_from`** – Path to the checkpoint for training resume

**`exp_dir`** – Folder name for saving experiment results

**`seed`** – Random seed

**`val_interval`** – Validation frequency in epochs

**`recognition_levels`** – Text levels to recognize: `word`, `line`, or `block` (training default: `["word"]`). Saved for inference; selecting a level does not create regions or change the model's training task.

**`pretrain_weights`** – `"default"` downloads the `trba_lite_g1` checkpoint; `null` trains from scratch. Local checkpoint paths and custom weight URLs are also supported. Resuming skips pretraining.

## Example training.json

The neighboring `config.json` contains the same example settings. These are
training options, not a ready-to-use inference config for existing weights.

This example supplies separate validation data and trains from scratch. Replace
the dataset and output paths before running it. Omit `resume_from` for a new
run; set it to an existing checkpoint path when resuming.

```json
{
    "train_csvs": [
        "path/to/train/labels1.csv",
        "path/to/train/labels2.csv"
    ],
    "train_roots": [
        "path/to/train/images1",
        "path/to/train/images2"
    ],
    "val_csvs": [
        "path/to/val/labels1.csv",
        "path/to/val/labels2.csv"
    ],
    "val_roots": [
        "path/to/val/images1",
        "path/to/val/images2"
    ],
    "charset_path": "path/to/charset.txt",
    "recognition_levels": ["word"],
    "pretrain_weights": null,
    "encoding": "utf-8",
    "img_h": 32,
    "img_w": 128,
    "max_len": 40,
    "hidden_size": 256,
    "batch_size": 64,
    "epochs": 100,
    "lr": 0.001,
    "optimizer": "Adam",
    "scheduler": "OneCycleLR",
    "weight_decay": 1e-4,
    "momentum": 0.9,
    "shift_limit": 0.05,
    "scale_limit": 0.05,
    "rotate_limit": 5,
    "p_ShiftScaleRotate": 0.5,
    "brightness_limit": 0.2,
    "contrast_limit": 0.2,
    "p_BrightnessContrast": 0.3,
    "invert_p": 0.02,
    "train_proportions": [0.6, 0.4],
    "val_size": 500,
    "num_workers": 4,
    "exp_dir": "path/to/experiments",
    "seed": 123,
    "val_interval": 5
}
```

## Example of a charset.txt file

Use one token per line. `<PAD>`, `<SOS>` and `<EOS>` are required and are looked
up by name. `<BLANK>` is optional and is used for CTC when present. Include a
literal space on its own line if labels contain spaces; it does not have to be
the fourth token. Empty lines are ignored.

```plaintext
<PAD>
<SOS>
<EOS>
 
a
b
c
0
1
2
А
Б
.
,
```

**Special tokens and spaces:**

* `<PAD>` – padding token
* `<SOS>` – start of sequence token
* `<EOS>` – end of sequence token
* `<BLANK>` – optional CTC blank token
* ` ` – literal space, needed to represent spaces in labels

**Characters for recognition:**

* Latin letters (a–z, A–Z)
* Digits (0–9)
* Cyrillic letters (а–я, А–Я)
* Punctuation marks and special symbols

For training from scratch, customize the alphabet to cover your labels. Keep
tokens unique. For existing weights, preserve the complete token order used
during training: each line defines a model output index. Adding, removing or
reordering tokens requires matching model weights.

## Weight freezing policies (fine-tuning)

* Pass these parameters to `TRBA.train(...)`, or put them in `training.json`
  and load it explicitly with `TRBA.train(**options)`:

  * `freeze_cnn`: "none" | "partial" | "full"
  * `freeze_enc_rnn`: "none" | "partial" | "full"
  * `freeze_attention`: "none" | "partial" | "full"

Semantics of **partial**:

* **CNN**: layers `conv0`, `layer1`, `layer2`, `layer3` are frozen; layers `layer4` and `conv_out` remain trainable.
* **enc_rnn**: the first BiLSTM block is frozen; remaining blocks stay trainable.
* **attention**: for the attention decoder, `attention_cell` is frozen and
  `generator` remains trainable. For PARSeq, all decoder parameters except
  `generator` are frozen. Full freezing freezes the entire decoder in either case.

Examples:

```python
from manuscript.recognizers import TRBA

summary = TRBA.train(
    train_csvs=["train.tsv"],
    train_roots=["train"],
    val_csvs=["val.tsv"],
    val_roots=["val"],
    freeze_cnn="partial",
    freeze_enc_rnn="none",
    freeze_attention="full",
)
```

These fields can also be specified in `training.json` and passed through
`TRBA.train(**options)` as shown above.
