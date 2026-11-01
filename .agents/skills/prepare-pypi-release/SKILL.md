---
name: prepare-pypi-release
description: Prepare a Manuscript OCR release for PyPI, reconcile repository versions, validate artifacts and optionally publish to PyPI and GitHub when requested. Use for release preparation, publication and recovery from partial uploads.
---

# Prepare a PyPI release

Use `pyproject.toml` as the package-version source of truth. Derive the target
version, remote and release commit from current state; do not hardcode 0.1.13.
Preparation alone does not authorize uploading, pushing, tagging or publishing.
When the user already requested these actions, proceed within that scope without
asking again. A PyPI-only request does not imply a GitHub release.

## Reconcile repository versions

Inspect the working tree, remotes, recent commits, tags and release workflows.
Preserve user edits. Record the exact release SHA and identify uncommitted files
that would enter an artifact. Do not publish artifacts differing from the
identified release source without resolving and disclosing that difference.

Check live PyPI JSON and GitHub releases/tags using available tools, CLI or API.
An existing version may be a partial upload. Do not bump versions merely to
evade an upload failure.

Create a compact version-consistency map: location, role, expected value, result.
Read actual contents, not just keyword matches. Check:

- `pyproject.toml`: package version and Python support.
- Runtime version exports, if present, and installed distribution metadata.
  Do not add `__version__` when the project uses distribution metadata instead.
- `docs/sphinx/conf.py`: default `release`/`version` and environment overrides.
- `.github/workflows/docs.yml`: `LATEST_DOCS_VERSION`, redirect HTML, latest
  version metadata, version list and links. Inspect every hardcoded copy.
- `docs/build_docs.bat` and other current build/publication configuration.
- Active release notes, migration destinations, current EN guides and Russian
  catalogs, README, browser instructions, notebook links and examples naming
  the current release/version/branch.
- Wheel/sdist filenames and internal `METADATA`/`PKG-INFO` versions.
- Requested tag (normally `v<version>`), GitHub release target SHA, title and
  assets, and PyPI version/file metadata.

Distinguish historical release sections, `docs/sphinx/_archive/`, supported-version
lists, model artifact tags and runtime compatibility ranges. Older values here
may be intentional: model downloads can use `v0.1.0` independently of the package.
Never globally replace versions or change compatibility bounds just to make them
equal. Verify historical links point to their intended source.

Fix confirmed current-release drift within preparation scope, preserving history.
Use [docs-update](../docs-update/SKILL.md) for affected translatable documentation;
use [docs-audit](../docs-audit/SKILL.md) if freshness has not been established.
Reuse applicable checks from the same task while their inputs remain unchanged.
State unresolved mismatches before publication.

## Validate the release commit

Use `.github/workflows/test_install.yml` as the reference. Check CI for the exact
release SHA, including wheel and integration jobs. Diagnose red CI before
publication; pending/unavailable CI is unverified. Run relevant local checks
for preparation changes and `git diff --check`, reporting environment limits.
If changes require a new commit/push, do so only when authorized and verify that
new release SHA. Do not tag an older SHA while packaging newer changes.

If all tests pass but pytest returns 1 with `--require-integration-deps`, inspect
`pytest_sessionfinish`: unexpected skips can fail this repository's session.
Unavailable optional CUDA/DLPack capabilities can justify explicit skips;
installed CPU/GPU distributions do not prove runtime capabilities. Do not weaken
dependency checks or hide failures in supported environments to get green CI.

## Build and check distributions

Use `env/bin/python` when available and a fresh task-specific output directory.
Prefer normal PEP 517 build isolation so build dependencies resolve correctly.
Do not modify `env/` or upgrade its dependencies merely to prepare a release.
If isolated dependencies are unavailable, state that limitation before choosing
a known-compatible alternative; do not silently fall back to an old backend.

```bash
release_dir=$(mktemp -d "${TMPDIR:-/tmp}/manuscript-release.XXXXXX")
env/bin/python -m build --outdir "$release_dir"
env/bin/python -m twine check "$release_dir"/*.whl "$release_dir"/*.tar.gz
```

Record absolute artifact paths. Require wheel and sdist to carry the target
version and normalized name. The sdist name must be
`manuscript_ocr-<version>.tar.gz`, not `manuscript-ocr-<version>.tar.gz`.
Old setuptools (observed with 65.5.0) can emit the latter; `twine check` alone
does not detect PyPI's filename rejection. Check sdist root and metadata too.
Prefer rebuilding with a suitable isolated backend. For recovery from a
filename-only rejection, a verified byte-identical copy under the canonical
filename is sufficient; do not alter source/metadata to disguise another build.

Inspect contents: Python sources, TRBA config/charset and
`manuscript/runtimes/bridge.mjs` must be present; environments, credentials,
caches and unintended development files must be absent. Check Python requirements
and optional extras in metadata. Build the wheel from the sdist (normal
`python -m build` behavior) to verify source packaging is sufficient.

Install the wheel into a temporary environment with required inference
dependencies, outside the checkout, and run `tests/package/wheel_smoke.py`.
Ensure imports resolve to the installed wheel, not `src/`; do not reinstall into
the user's existing environment. The smoke check covers resources, ONNX inference
and inference without loading Torch/Transformers. Record SHA-256 hashes of the
validated files; upload those files rather than rebuilding before publication.

## Publish or recover a partial upload

Use the existing publication path. Do not introduce Trusted Publishing, new
credentials or a release workflow without a corresponding request. Never print
saved credentials or ask for tokens in chat.

For local Twine use configured credentials or the user's interactive terminal.
`This environment is not supported for trusted publishing` is not itself a
failed upload when API-token authentication is used. If the agent cannot access
a credential, finish preparation and give an exact local command with explicit
artifact paths; do not claim publication succeeded.

On HTTP 400 or another upload failure:

1. Query live PyPI version JSON. Compare remote filenames and SHA-256 hashes
   with prepared artifacts; an earlier file may have succeeded.
2. Inspect the server response (`--verbose` if needed, keeping credentials out
   of shared output). Do not assume every HTTP 400 is a filename problem.
3. Retry only missing corrected artifacts. Do not blindly re-upload accepted
   files or treat `--skip-existing` as proof of identity. PyPI files cannot be
   replaced; different content requires a deliberate version decision. Do not
   delete/recreate releases as a retry strategy.
4. After a corrected retry, stop on another unexplained rejection and report
   the response and current remote state rather than loop or bump versions.

For an authorized GitHub release, derive notes from active release notes, include
migration/install guidance, target the validated SHA and attach the same checked
distributions. Do not blindly overwrite tags/assets. Reconcile corrected artifact
names explicitly while preserving unrelated model assets and historical tags.

## Verify and report

Re-query PyPI/GitHub after publication. Check target version/SHA and remote file
hashes/sizes against prepared artifacts. Report wheel, sdist and GitHub publication
separately; an upload progress bar is not confirmation. Give release URLs,
validation results and remaining blockers. For preparation-only tasks provide
artifact paths and outstanding steps. State whether repository version consistency
is confirmed, corrected or unverified and explain intentionally older references.
