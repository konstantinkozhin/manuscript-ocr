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

## TRBA Region Preparation

`TRBA` now keeps the current default behavior while allowing crop-preparation
customization when needed.

Default behavior is unchanged:

- `region_preparer="bbox"` extracts axis-aligned bounding boxes
- `rotate_threshold=1.5` auto-rotates tall crops before recognition
- `min_text_size=5` skips tiny detections

Built-in preparer presets:

- `"bbox"`: legacy axis-aligned crop behavior
- `"polygon_mask"`: tight crop with pixels outside the polygon masked to white
- `"quad_warp"`: perspective rectification for 4-point polygons, with bbox fallback

```python
from manuscript.recognizers import TRBA

recognizer = TRBA(region_preparer="bbox")
recognizer = TRBA(region_preparer="polygon_mask")
recognizer = TRBA(region_preparer="quad_warp")
recognizer = TRBA(
    region_preparer="bbox",
    region_preparer_options={"pad": 2},
)
```

`region_preparer_options` is reserved for built-in preset configuration. Current
options are:

- `"bbox"` / `"polygon_mask"`: `pad`
- `"polygon_mask"`: `background`
- `"quad_warp"`: `output_size=(width, height)`, `fallback_to_bbox`

For advanced cases, you can inject hooks into `TRBA` instead of writing a full
custom recognizer:

```python
from functools import partial
import numpy as np

def my_preparer(page, image, recognizer=None, options=None):
    regions = []
    for block in page.blocks:
        for line in block.lines:
            for word in line.words:
                poly = np.asarray(word.polygon, dtype=np.float32)
                crop = image[10:40, 10:80]
                regions.append({"word": word, "image": crop, "polygon": poly})
    return regions

recognizer = TRBA(region_preparer=partial(my_preparer))
```

If you want complete control over recognition logic, the simplest route is
still to provide your own recognizer class with:

```python
def predict(self, page: Page, image: Optional[np.ndarray] = None) -> Page:
    ...
```

## Intermediate Snapshots

After each `predict` call:

- `pipeline.last_detection_page`
- `pipeline.last_layout_page`
- `pipeline.last_recognition_page`
- `pipeline.last_correction_page`

Skipped stages keep corresponding `last_*` property as `None`.
