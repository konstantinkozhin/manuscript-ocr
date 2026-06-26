import torch
from pathlib import Path

from manuscript.detectors._pse import PSE

# Для быстрого теста используем test split и как train, и как val
train_images = r"C:\shared\data02065\school_notebooks_RU\train_images"
train_annotations = r"C:\shared\data02065\school_notebooks_RU\train.json"
val_images = r"C:\shared\data02065\school_notebooks_RU\test_images"
val_annotations = r"C:\shared\data02065\school_notebooks_RU\test.json"

experiment_root = "experiments"
model_name = "resnet_pse"
resume_checkpoint = (
    Path(experiment_root) / model_name / "checkpoints" / "last_state.pt"
)

PSE.train(
    train_images=train_images,
    train_anns=train_annotations,
    val_images=val_images,
    val_anns=val_annotations,
    experiment_root=experiment_root,
    model_name=model_name,
    epochs=300,
    batch_size=1,
    target_size=1440,
    val_interval=3,
    lr_scheduler="none",
    # Continue from the last fully completed epoch when a state checkpoint exists.
    resume_from=resume_checkpoint if resume_checkpoint.exists() else None,
    device=torch.device("cuda"),
)
