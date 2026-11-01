---
name: docs-update
description: Update Manuscript OCR guides, API documentation, release notes and Russian translations to match code changes or audit findings. Use when updates are requested or required by an authorized implementation task.
---

# Update documentation and translations

## Scope and change control

Identify behavior from the relevant diff, implementation and tests. When audit
findings are supplied, use them as the initial scope and verify them against the
current implementation. Do not repeat a full repository audit when those findings
establish the required changes; inspect related documentation and translations.
If findings are insufficient, investigate the missing evidence within scope.

Limit edits to the requested behavior, affected documentation and corresponding
translations. Inspect the working tree before running documentation tooling so
its changes can be distinguished from pre-existing user edits. Preserve unrelated
content and user changes throughout the update.

If a discrepancy may be an implementation defect against an established
contract, report it and establish the intended behavior from available evidence.
Do not change executable behavior or document a suspected bug as the intended
API merely to eliminate a mismatch. Continue independent confirmed corrections.

## Update sources

- Update public API docstrings in `src/manuscript` and Sphinx directives in
  `docs/sphinx/api/` when needed.
- Update English guides in `docs/sphinx/*.rst` for installation, usage and runtime
  requirements. Add migration steps when existing integrations must change.
- Update the active release's `release_notes.rst` section for relevant
  user-visible behavior changes. Documentation-only corrections do not normally
  need release notes, unless they clarify a significant incorrect compatibility
  claim. Do not manufacture a release entry for a typo or routine example fix.
- Update the Russian `README.md` when installation or quick-start behavior
  changes. Check `docs/WEB_RUNTIME.md` and `scripts/browser_probe/README.md`
  for browser setup changes.

Describe defaults, constraints and failure/fallback behavior precisely. Optional
package installation alone does not guarantee runtime capabilities. Keep examples
consistent with current constructors, results and dependencies. Internal fixes
need documentation only when they affect user-facing claims; not every test
change needs a release note.

Preserve each source's language. English guides and docstrings stay English;
Russian translations belong in Russian catalogs. Do not translate unrelated
repository instructions or comments as a side effect.

## Synchronize Russian translations

Apply this section when translatable Sphinx sources or catalogs are affected.
A README-only correction does not require a gettext merge or MO compilation.

Use the documentation dependencies and tooling in `.github/workflows/docs.yml`.
Typical commands, using the project's Python environment:

```bash
# Run from the repository root. Keep this variable for the builds below.
docs_check_dir=$(mktemp -d "${TMPDIR:-/tmp}/manuscript-docs.XXXXXX")
python -m sphinx -E -a -b gettext -D language=en docs/sphinx "$docs_check_dir/gettext"
sphinx-intl update -p "$docs_check_dir/gettext" -d docs/sphinx/locale -l ru
```

Use a new output directory per task: old directories can retain templates for
removed pages. Regenerate templates if English sources or extracted docstrings
change after generation. Do not merge templates whose generation failed without
checking which outputs are complete.

Review new, changed and fuzzy messages against fresh templates. Translate affected
prose faithfully, including defaults, limitations and warnings. Preserve API names,
code literals, placeholders and RST roles/targets. Clear fuzzy flags only after
reviewing translations. After updating catalogs, inspect the PO diff and remove
unrelated automatic changes without discarding pre-existing user modifications.
Do not normalize or reformat entire catalogs unless requested.

Compile only affected `.mo` catalogs when feasible: they are tracked in this
repository. Verify their messages match current PO translations, not just that
compilation succeeded. Changed MO files should correspond to affected PO catalogs
or a confirmed stale binary within scope. Exclude unrelated tracked MO changes,
including those generated implicitly by Sphinx. Do not edit historical
`docs/sphinx/_archive/` pages to describe current behavior. Generated `_build/`
and `_site/` output does not replace source documentation.

## Validation and completion

Choose checks for the affected surfaces. For Sphinx/API/translation changes use
the builds below; a change confined to a standalone Markdown file needs checks
of that file's content and links rather than unrelated Sphinx rebuilds.

```bash
python -m sphinx -W --keep-going -b html -D language=en docs/sphinx "$docs_check_dir/en"
python -m sphinx -W --keep-going -b html -D language=ru docs/sphinx "$docs_check_dir/ru"
git diff --check
```

Inspect affected rendered sections in both languages, including examples and
cross-references. Verify examples against actual public APIs; rendering alone
does not validate their behavior. Verify expected API objects are present: this
project mocks optional dependencies and suppresses `autodoc.import_object`
warnings, which can hide missing class documentation in a successful build.
Compare catalogs with fresh templates for missing or fuzzy messages within
scope. Validate catalog syntax and placeholders; investigate
format-detector false positives in prose without distorting its meaning.

Reuse validation already completed in this task only if its relevant inputs are
unchanged. Perform final builds after the source and translation corrections;
do not repeat a full audit or identical build solely because another skill
requested the same check.

If external intersphinx inventories are unavailable, report the limitation.
An offline build may disable `intersphinx_mapping` through the Sphinx Python API.
Do not change publication configuration or suppress content warnings to pass.

Classify validation failures as introduced by this update, pre-existing,
environment-related or undetermined, using evidence where available. Fix failures
introduced by the update. Continue useful checks when another check is blocked;
do not expand into unrelated repairs solely to obtain a green build.

Review the final diff for unintended edits, unrelated PO/MO changes, stale
examples or references, accidental generated-file changes and loss of user edits.
Finish when scoped corrections and translations are complete, affected binaries
are consistent, and validation has passed or remaining limitations are explicit.
Unfixed failures introduced by the update leave the affected work incomplete;
environment-blocked checks must remain labeled unverified.

Report updated topics and languages, validation results, remaining issues and
checks that could not be completed. Do not claim successful validation when
required checks failed or were skipped. State publication status; editing or
building documentation does not authorize deployment.
