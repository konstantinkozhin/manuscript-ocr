Политики заморозки весов (fine-tuning)

- Параметры принимаются в `TRBAInfer.train(...)` и/или в `config.json`:
  - `freeze_cnn`: "none" | "partial" | "full"
  - `freeze_enc_rnn`: "none" | "partial" | "full"
  - `freeze_attention`: "none" | "partial" | "full"

Семантика partial:
- CNN: замораживаются `conv0`, `layer1`, `layer2`, `layer3`, оставляются обучаемыми `layer4` и `conv_out`.
- enc_rnn: замораживается первый BiLSTM блок, последний — обучаемый.
- attention: замораживается `attention_cell`, обучаемым остаётся `generator`.

Примеры:

```python
from manuscript import TRBAInfer

summary = TRBAInfer.train(
    train_csvs=["train.tsv"],
    train_roots=["train"],
    val_csvs=["val.tsv"],
    val_roots=["val"],
    freeze_cnn="partial",
    freeze_enc_rnn="none",
    freeze_attention="full",
)
```

Также можно указать эти поля в `config.json` при запуске обучения через конфиг.

