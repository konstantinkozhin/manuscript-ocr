---
name: create-github-release
description: Prepare or publish a Manuscript OCR GitHub release using existing package releases as examples, including release notes, EN/RU HTML documentation, checksums and requested package artifacts. Use for GitHub release creation, updates or interrupted publication.
---

# Create a GitHub release

Prepare the release from the requested version and exact source commit. Creating
this skill or preparing artifacts does not authorize publishing. If the user
already requested publication, proceed within that authorization without another
confirmation. Do not infer PyPI upload, model publication or edits to historical
releases from a request for a GitHub package release.

## Learn the repository's release conventions

Inspect current remotes, working tree, tags and live releases through available
connected tools, `gh` or the GitHub API. Preserve user changes. Read recent
package release bodies and asset lists; do not infer conventions from tags alone.

Two reference package releases inspected when this skill was created:

- [v0.1.12](https://github.com/konstantinkozhin/manuscript-ocr/releases/tag/v0.1.12):
  title `manuscript-ocr 0.1.12`, English change bullets, instructions to unpack
  `manuscript-ocr-0.1.12-docs.zip` and open `index.html`, a separate model-release
  link, and `SHA256SUMS.txt` alongside the documentation ZIP.
- [v0.1.13](https://github.com/konstantinkozhin/manuscript-ocr/releases/tag/v0.1.13):
  title `Manuscript OCR 0.1.13`, English overview, Highlights and Migration,
  version-pinned installation commands, links to migration/full release notes,
  wheel and sdist assets. At inspection it had no docs ZIP/checksum file and
  its sdist used an old noncanonical filename; do not copy these omissions or
  that filename as requirements.

[v0.1.0](https://github.com/konstantinkozhin/manuscript-ocr/releases/tag/v0.1.0)
is a separate model-asset release, not a package-release template. Preserve its
weights, configurations, external ONNX data, licenses and download URLs. Do not
copy large model files into every package release.

Re-read the live examples when applying the skill: these observations can change.
Use the current user request to decide scope if it differs from earlier releases.

## Fix the version and source

Read the target version from `pyproject.toml` unless explicitly supplied. Check
package metadata, current documentation version in `docs/sphinx/conf.py`,
`docs/build_docs.bat`, `.github/workflows/docs.yml` (including hardcoded redirects
and version JSON), and active release notes agree. Historical notes, archived
docs, model tags and compatibility bounds may correctly name older versions;
do not replace them globally. Use [docs-update](../docs-update/SKILL.md) for
confirmed documentation/translation changes.

Identify the remote source SHA. Check CI for that SHA, using
`.github/workflows/test_install.yml` as the reference. Pending or inaccessible CI
is unverified, not green. Diagnose failures before publication. If changes are
needed, do not publish old tagged code with new working-tree artifacts. Commit
and push only within authorization, then validate the updated SHA.

Use `v<version>` unless the user/repository specifies otherwise. Resolve any
existing tag to its commit, dereferencing annotated tags; `target_commitish`
alone does not prove what an existing tag points to. Never force-move a tag to
match a different build. Report conflicts for a deliberate version/source decision.

## Prepare reviewable notes and assets

Write English notes consistent with the inspected package releases. Describe
concrete user-facing changes from the previous package release to the target SHA,
using `docs/sphinx/release_notes.rst` and the actual diff. Include migration
requirements and installation extras when relevant. Preserve the distinction
between native inference, optional training and experimental browser execution.
Prefer immutable release-tag links for migration and full release notes. Link
model weights separately and explain they are not bundled. Do not claim PyPI
availability until verified live.

For an ordinary full package release, prepare the offline EN/RU documentation
ZIP and checksums as in v0.1.12; include wheel/sdist when requested or part of
the selected release convention. For an explicitly notes-only or narrower update,
respect that scope. State the selected assets before publication.

Build HTML from the release source in a temporary copy so Sphinx cannot rewrite
tracked MO files. Follow `.github/workflows/docs.yml` and
[docs-audit](../docs-audit/SKILL.md) for warning/error handling and verification.
Reuse validated HTML only when source SHA, documentation/catalog inputs and
release configuration match. Set release/version to the target version.

Package `manuscript-ocr-<version>-docs.zip` with a root `index.html` linking to
`en/index.html` and `ru/index.html`, plus all required local HTML/static/image
assets. Include no secrets, caches, doctrees, source environments or unrelated
historical builds. Extract the ZIP into a fresh directory and verify both
language entrypoints, local navigation/assets and displayed version. Report
network-dependent diagrams, external links or browser restrictions honestly;
do not promise offline functionality that was not verified.

If wheel/sdist are included, use the build/validation guidance in
[prepare-pypi-release](../prepare-pypi-release/SKILL.md), without invoking its
publication steps. Reuse already verified files rather than rebuilding. Check
metadata version, required resources and installed-wheel smoke coverage. The
canonical sdist filename is `manuscript_ocr-<version>.tar.gz`.
When the version exists on PyPI, compare artifact SHA-256 hashes to PyPI JSON;
do not attach a different same-version wheel without resolving the conflict.

Generate `SHA256SUMS.txt` over all assets selected for this release except the
checksum file itself, using basename-only entries. Verify it against the final
files. GitHub's automatic source ZIP/tar links are generated from the tag and are
not substitutes for a validated Python sdist or offline documentation archive.

## Publish or update within authorization

Use an available GitHub connector, otherwise authenticated CLI/API. Do not install
a new integration when existing access suffices. Never expose credentials.
Pass multiline notes as a structured body or `--notes-file`, not shell-interpolated
text. Set tag, validated target SHA, title, body and draft/prerelease/latest state
explicitly; do not mark a prerelease as the latest stable release.

For an authorized publication a draft may stage notes/assets, but the requested
published release is unfinished until it is made public. Do not silently leave
it as a draft. If the user requested a draft, preserve that state.

If creation/upload is interrupted, query the existing release and asset list
before retrying. Reuse matching assets after checking hashes (download them if
GitHub provides no digest). Upload only missing files. Do not use blind clobber,
delete/recreate the release, replace unrelated assets or retag as a retry strategy.
Stop on conflicting content or another unexplained rejection and report what
already succeeded and what remains. Any necessary replacement must be explicit
within the user's update scope.

## Verify completion

Read the live release again: URL, published/draft state, prerelease/latest
selection, resolved tag SHA, notes and expected asset names/sizes/hashes. Download
and verify checksum contents when feasible. Confirm version consistency across
source, tag, notes, documentation archive and included packages. Distinguish
GitHub publication from PyPI publication and documentation-site deployment;
perform the latter only when requested.

Report the release link, version/SHA, assets, checks and unresolved limitations.
For preparation-only work, provide local notes/artifact paths and publication
status. Do not claim success from an upload progress bar or an unverified API call.
