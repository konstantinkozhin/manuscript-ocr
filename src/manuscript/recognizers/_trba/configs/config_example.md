# TRBA Model Configuration

## Parameters for TRBA (required)

**`img_h`** – Height of the input image for the model in pixels

**`img_w`** – Width of the input image for the model in pixels

**`hidden_size`** – Hidden layer size in the RNN part of the model

## Training parameters (required for training)

**`charset_path`** – Path to the file containing the character set

## Additional parameters

**`max_len`** – Maximum length of the recognized character sequence

**`encoding`** – Text data encoding

## Training parameters

**`train_csvs`** – Paths to CSV files with labels for training

**`train_roots`** – Root folders with images for training

**`val_csvs`** – Paths to CSV files with validation labels

**`val_roots`** – Root folders with images for validation

**`batch_size`** – Batch size for training

**`epochs`** – Maximum number of training epochs

**`lr`** – Learning rate

**`optimizer`** – Optimization algorithm ("Adam", "SGD", "AdamW")

**`scheduler`** – Learning rate scheduler

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

**`train_proportions`** – Mixing proportions for multiple datasets

**`val_size`** – Number of validation samples

**`num_workers`** – Number of worker processes for data loading

**`resume_path`** – Path to the checkpoint for training resume

**`exp_dir`** – Folder name for saving experiment results

**`seed`** – Random seed

**`eval_every`** – Validation frequency in epochs

## Example of a full config.json file

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
    "encoding": "utf-8",
    "img_h": 32,
    "img_w": 128,
    "max_len": 40,
    "hidden_size": 256,
    "batch_size": 64,
    "epochs": 100,
    "lr": 0.001,
    "optimizer": "Adam",
    "scheduler": "StepLR",
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
    "resume_path": "path/to/checkpoint.pth",
    "exp_dir": "path/to/experiments",
    "seed": 123,
    "eval_every": 5
}
```

## Example of a charset.txt file

**Important:** The first 4 lines are mandatory and must appear exactly in this order, then you may add any characters you need.

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

**Mandatory special tokens:**

* `<PAD>` – padding token
* `<SOS>` – start of sequence token
* `<EOS>` – end of sequence token
* ` ` – space (4th line, required)

**Characters for recognition:**

* Latin letters (a–z, A–Z)
* Digits (0–9)
* Cyrillic letters (а–я, А–Я)
* Punctuation marks and special symbols

You may customize the alphabet by adding or removing characters after the first 4 required lines.

# Weight Freezing Policies (fine-tuning)

* Parameters are accepted in `TRBA.train(...)` and/or in `config.json`:

  * `freeze_cnn`: "none" | "partial" | "full"
  * `freeze_enc_rnn`: "none" | "partial" | "full"
  * `freeze_attention`: "none" | "partial" | "full"

Semantics of **partial**:

* **CNN**: layers `conv0`, `layer1`, `layer2`, `layer3` are frozen; layers `layer4` and `conv_out` remain trainable.
* **enc_rnn**: the first BiLSTM block is frozen, the last one is trainable.
* **attention**: `attention_cell` is frozen, `generator` remains trainable.

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

These fields can also be specified inside `config.json` when training is launched using a config file.