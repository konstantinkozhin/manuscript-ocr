# Model registry

The official `registry.json` lives at the root of `main`. The library code is
developed on `v_0_1_13`. GitHub is tried first and GitVerse second. GitVerse may
temporarily have no registry. The package includes a fallback snapshot in
`src/manuscript/models/registry.json`; update it when preparing a release.

No request is made on import. A model alias uses the cached registry. A missing
registry or unknown alias triggers an update. When all artifact mirrors fail,
the registry is refreshed and downloading is retried once. There is no TTL or
background polling. A malformed response never replaces the valid cached file.

```python
from manuscript import models
from manuscript.recognizers import TRBA

recognizer = TRBA(weights="trba_lite_g2")
models.info("trba_lite_g2")
models.download("trba_lite_g2")
models.download("trba_lite_g2", artifact="checkpoint")
models.refresh()
models.add_registry("https://example.org/registry.json", priority=True, persist=True)
models.refresh()  # explicitly activate a changed source configuration
models.path("trba_lite_g2")
models.verify("trba_lite_g2")
models.is_installed("trba_lite_g2")
models.list_installed()
models.remove("trba_lite_g2")  # current registered version only
```

Extra catalogs supplement the official catalog. `priority=True` allows their
entries to override official names; otherwise official names win. Prioritizing
a catalog means trusting its metadata and artifact URLs. `persist=True` stores
the setting for subsequent processes. Missing custom sources retain their last
valid cached catalog. The two official URLs are mirrors of a single catalog.

## Storage

Default root: `~/.manuscript`; override it with `MANUSCRIPT_HOME`.

```text
.manuscript/
  registries/
    registry.json            # effective catalog
    sources.json             # persistent custom sources
    <source-hash>.json        # last valid catalog per source/group
  models/
    trba_lite_g2/
      <model-version>/
        model.json           # local copy of the model entry
        trba_lite_g2.onnx
        trba_lite_g2.json
        trba_lite_g2.txt
        LICENSE.txt          # when a license artifact becomes available
```

For unknown model versions, the directory is `unversioned-<content-fingerprint>`.
Mirror changes do not change that fingerprint. Required artifacts are downloaded
together. Checkpoints and lexicons can be optional; CharLM requests its lexicon
when needed. A declared license is always downloaded alongside the requested
artifacts. `null` license artifacts are skipped until a real file is published.
`list_installed()` includes partial bundles; `is_installed()` verifies required
files. Direct URLs and local paths remain supported through the existing API.

Files download to unique temporary files in the model directory. Size and
SHA-256, when known, are checked before atomic replacement. Old flat files under
`weights/` are copied only if their SHA-256 matches; old files are never removed.

## Editing registry.json

`schema_version` is currently `1`. The `models` mapping uses the public alias as
its key. Each entry contains:

- `model_classes`: allowed public class IDs (`TRBA`, `EAST`, `YOLO`, `CharLM`,
  `PPOCRv5Rec`). These IDs never trigger dynamic imports.
- `model_version`: string or `null`; use a new version when changing artifacts.
- `library_version`: a Python version specifier such as `>=0.1.13,<0.2`, or `null`
  when compatibility has not been established.
- `task`, `description`, `architecture`, `license`: descriptive metadata.
- `artifacts`: roles such as `weights`, `config`, `charset`, `vocab`, `lexicon`,
  `checkpoint`, `license`. An unknown artifact can be `null`.

Each available artifact has `filename`, `format`, `size`, `sha256`, `urls` and
optionally `required` (defaults to `true`). URLs are tried in listed order.
Use actual download URLs, not HTML pages. `size` and `sha256` can temporarily be
`null`; their corresponding integrity checks then cannot be performed. Unknown
metadata uses JSON `null`, never `NaN` or Python `None`.

All mirrors of an artifact must serve identical bytes. Changing only `urls`
is sufficient to move hosting. Add a license as an ordinary artifact named
`LICENSE.txt`; its legal identifier/description can also appear in the entry's
`license` metadata. Do not infer a weights license from the library license.

The initial catalog contains all 31 assets of GitHub release `v0.1.0`, grouped
into 10 models, including the checkpoint-only `trba_base_g0`. Request that model
with `artifact="checkpoint"`; it has no published ONNX inference weights.
