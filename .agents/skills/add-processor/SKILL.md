---
name: add-processor
description: Add a Manuscript OCR recognizer, detector, corrector or layout class using existing processor contracts, package conventions and API documentation. Use for new processor implementations, with minimal tests and a final documentation audit and update.
---

# Add a processor class

Implement the requested class in the same style as its closest existing peers.
Keep scope to the processor, necessary support code, minimal tests and its
documentation. Do not create a new framework around a single implementation.

## Inspect the relevant contract

Read `AGENTS.md`, the appropriate base in `src/manuscript/api/`, and a current
implementation with similar inputs and runtime needs before choosing hooks or
signatures. Inspect the shared helpers and relevant contract tests. Prefer the
current base contract over copying an older peer's workaround.

| Kind | Implementation directory | Base module | API page | Useful peers |
| --- | --- | --- | --- | --- |
| Detector | `src/manuscript/detectors/` | `api/detector.py` | `api/detectors.rst` | `_yolo`, `_ppocr`, `_rfdetr`, `_mask2former` |
| Recognizer | `src/manuscript/recognizers/` | `api/recognizer.py` | `api/recognizers.rst` | `_ppocr_rec`, `_trba`, `_trocr` |
| Corrector | `src/manuscript/correctors/` | `api/corrector.py` | `api/correctors.rst` | `_charlm` |
| Layout | `src/manuscript/layouts/` | `api/layout.py` | `api/layouts.rst` | `_simplesorting`, `_readingsorting` |

Base modules are relative to `src/manuscript/`; API pages are relative to
`docs/sphinx/`. Read the applicable `BaseDetector`, `BaseRecognizer`,
`BaseCorrector` or `BaseLayout`, together with its inherited lifecycle.

Confirm model inputs/outputs, preprocessing, configuration and dependencies from
the supplied artifacts or authoritative specifications. Do not invent tensor
layouts, label mappings, tokenizers, model URLs or working presets. If essential
information is unavailable, ask for it while progressing independent work.

## Implement within existing conventions

- Put the implementation under the appropriate `src/manuscript/<kind>/_<name>/`
  package, following peers' file organization. Add helper files only when the
  implementation needs them; avoid README files, placeholder configs, sample
  assets and duplicate utilities.
- Reuse existing artifact loading, runtime/session handling, configuration,
  geometry and tree helpers. Implement only the required inference hooks rather
  than duplicating a base class's public prediction pipeline.
- A detector returns a valid `Page` in original-image coordinates and dimensions.
  Follow `BaseDetector` class mapping and optional raw/mask/output return contracts
  where applicable. Do not fabricate masks, confidences or model outputs.
- A recognizer follows `BaseRecognizer` region preparation, selected recognition
  levels, prediction normalization and result assignment. Preserve region order,
  geometry and tree structure in the returned copy, leaving the input Page
  unchanged. Respect batching and crop options supported by the base. Inspect
  current hooks rather than assuming their names.
- A corrector follows `BaseCorrector` selected levels and copy semantics; preserve
  geometry and unrelated text and nodes.
- A layout accepts a `Page` and returns a valid `Page`; child order defines
  reading order. Preserve node identity, contents and geometry where the layout
  contract requires it. Reuse supplied detector results when appropriate rather
  than silently running detection again.
- Preserve the applicable lifecycle contract, including close/reuse and session
  initialization. Keep training/export dependencies out of ordinary inference
  imports. Do not claim runtime support that was not implemented or verified.

Stateless layouts and processors must not acquire fake weights or ONNX sessions
just because their base is artifact-oriented; follow stateless peers' constructor
and no-op session initialization. Model-backed implementations should use managed
runtime session creation and failure cleanup, and validate configuration before
expensive artifact loading where possible. Keep new optional dependencies out of
the base installation; check `pyproject.toml` when a new extra is actually needed.

Follow the package's existing public exposure conventions when public import is
part of the requested class: model packages often use lazy `__getattr__` and
`__all__`, while layouts currently use direct imports. Do not standardize other
classes as a side effect. If the intended exposure is unclear, follow the closest
peer and state the resulting import path.

Do not connect the class to demos, apps, notebooks, default Pipelines or automatic
model selection unless explicitly requested. Public import exposure and API
documentation do not authorize application integration. Add registry presets only
when requested and backed by real artifacts; never invent registry entries.

If a model export utility is needed for the requested implementation, place it in
`scripts/export_<name>.py`, following existing export scripts. Do not create export
machinery for a class that does not need it, or introduce training support unless
requested. Document the actual exported model contract and dependencies.

## Document the class in the established format

Write English API docstrings consistent with the relevant peers and Sphinx
configuration: a concise purpose, actual constructor parameters and defaults,
return behavior, important constraints and a small usage example where helpful.
Use the existing NumPy-style sections where applicable. Document custom public
methods; do not duplicate inherited documentation without a behavior difference.

Add the class to the API page listed above using existing
`autoclass` / `automethod` conventions and its supported import path. Explain
required configuration and installation choices in the existing guides when
needed, without creating a standalone README for the new class.

## Minimal verification

Choose the smallest meaningful checks for the new behavior: public import when
exposed, one representative prediction through the actual class, and the critical
contract or invalid-input case not already covered by shared tests. Reuse fixtures
and small fake sessions or local synthetic models where appropriate. Do not add
large parameter matrices, duplicate base-class tests or a new test framework.
This is a risk-based selection, not a fixed test quota: select relevant cases
such as coordinate restoration, an empty Page or a partial final batch only
when the new implementation introduces that risk. Check actual postconditions
and input-copy semantics, not just that `predict()` returns without raising.

Run focused tests and relevant existing contract checks. Use real-model checks
only when suitable artifacts and dependencies are available; report their absence
instead of treating a mock prediction as verified model compatibility.

## Required documentation finish

After implementation and focused tests, apply
[docs-audit](../docs-audit/SKILL.md) to the new class, its public contracts and
affected guides and translations. Then apply
[docs-update](../docs-update/SKILL.md) to resolve confirmed gaps and synchronize
the affected Russian PO/MO catalogs. If no update is needed, say why. Do not
expand this into a full repository audit.

The initial audit may defer HTML builds to the update's final validation.
After corrections, recheck the identified gaps and expected API content, including
EN/RU builds where feasible. Reuse unchanged validation results instead of running
the same builds again; a second full audit is unnecessary. Report blocked
validation and unresolved questions explicitly;
do not claim completion of a skipped documentation check.

Review the final diff for unnecessary files, accidental app integration and
unrelated changes. Report the class/import path, implemented contract, focused
test results, documentation/translation status and any unverified model/runtime
support. Do not publish documentation or change demos as part of this finish.
