# Pipeline API Compatibility

Pipeline passes a single `Page` object through configurable stages:

`detector -> layout -> recognizer -> corrector`

Default `Pipeline()`:

- detector: `EAST()`
- layout: `SimpleSorting()`
- recognizer: `TRBA()`
- corrector: `None`

## Stage Contracts

### Detector

```python
def predict(self, image) -> Dict[str, Any]:
    return {"page": page}
```

### Layout

```python
def predict(self, page: Page, image: Optional[np.ndarray] = None) -> Page:
    ...
```

### Recognizer

```python
def predict(self, page: Page, image: Optional[np.ndarray] = None) -> Page:
    ...
```

### Corrector

```python
def predict(self, page: Page, image: Optional[np.ndarray] = None) -> Page:
    ...
```

## Pipeline Configuration

```python
from manuscript import Pipeline

pipeline = Pipeline(
    detector=...,            # required stage (cannot be None)
    layout=...,              # optional, default SimpleSorting()
    recognizer=...,          # optional, default TRBA()
    corrector=None,          # optional, default None
    layout_after="detector", # detector | recognizer | corrector
)
```

### Disable Stages

```python
# Detection + layout only
Pipeline(recognizer=None, corrector=None)

# Detection + recognition only
Pipeline(layout=None, corrector=None)
```

If `layout_after` points to a disabled stage, layout still runs in that slot.

## Built-in Components Example

```python
from manuscript import Pipeline
from manuscript.detectors import EAST
from manuscript.layouts import SimpleSorting
from manuscript.recognizers import TRBA
from manuscript.correctors import CharLM

pipeline = Pipeline(
    detector=EAST(weights="east_50_g1"),
    layout=SimpleSorting(max_splits=10, use_columns=True),
    recognizer=TRBA(weights="trba_lite_g1", device="cuda", min_text_size=5),
    corrector=CharLM(),
    layout_after="detector",
)

result = pipeline.predict("document.jpg")
text = pipeline.get_text(result["page"])
```

## Runtime Options

```python
result, vis_img = pipeline.predict("document.jpg", vis=True)
result = pipeline.predict("document.jpg", profile=True)
```

## Intermediate Snapshots

After each `predict` call:

- `pipeline.last_detection_page`
- `pipeline.last_layout_page`
- `pipeline.last_recognition_page`
- `pipeline.last_correction_page`

Skipped stages keep corresponding `last_*` property as `None`.
