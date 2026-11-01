---
name: docs-audit
description: Check whether Manuscript OCR documentation and Russian translations match current code or a specified set of changes. Use for freshness reviews; report findings without editing unless updates are requested.
---

# Audit documentation freshness

Establish the scope: a diff, release, feature, or current repository. If no
baseline is supplied, inspect the working tree and recent commits. Identify
relevant documentation changes and trace subsequent user-visible code changes;
do not assume the latest documentation commit represents a complete sync.
If history does not provide a useful baseline, audit current behavior in the
requested area and state that change-history coverage is unavailable.

Prioritize installation, public API changes, defaults and migration requirements
before lower-impact material. Keep a compact mapping of relevant changes or
behaviors to documentation and verification status. State what was reviewed;
a keyword search alone cannot establish completeness.

For a whole-repository audit, inspect current public APIs and guides as well as
recent changes: long-standing undocumented features may not appear in the diff.
If coverage is partial, identify the remaining areas instead of narrowing the
request silently.

## Compare behavior and documentation

Use implementation, tests and package metadata as evidence. Trace user-visible
changes to their documentation: signatures, defaults, return values, exceptions,
installation extras, runtime requirements, serialization and migration steps.
Check examples against actual APIs. Distinguish implemented behavior from
behavior merely asserted in a docstring or test.

If implementation contradicts an established contract or intended behavior,
report the conflict as a possible implementation defect. Do not assume every
mismatch should be resolved by rewriting documentation to match the code.

Repository sources:

- `README.md`: Russian installation and quick start.
- `docs/sphinx/*.rst`: English guides, release notes and migration instructions.
- `docs/sphinx/api/`: API directives, with content extracted from docstrings in
  `src/manuscript`.
- `docs/WEB_RUNTIME.md` and `scripts/browser_probe/README.md`: browser setup.
- `pyproject.toml` and `.github/workflows/`: dependencies and validation commands.

Generated API coverage does not replace a guide when users need configuration,
installation choices or migration steps. Historical release notes describe their
release; a dependency removed later does not make an older note incorrect.

## Check translations

Check the language surfaces within scope. A review confined to a standalone
Markdown file does not require rebuilding unrelated Sphinx catalogs.

Generate English Sphinx gettext templates in a new, task-specific temporary
directory. Reusing an old output directory can retain POT files for removed
pages even with a fresh Sphinx environment. Compare their
message IDs with `docs/sphinx/locale/ru/LC_MESSAGES/**/*.po`, including messages
from current docstrings. Check missing IDs, empty strings, fuzzy entries,
changed defaults and meaning, and preservation of code and RST references.
Nonempty translations do not prove semantic correctness.

Compare parsed gettext messages, including context and plural forms where
present, rather than raw PO lines. Ignore line wrapping, source-location churn
and obsolete IDs absent from fresh templates. A changed ID is a review candidate,
not automatically a missing translation: check whether the new message already
has a valid translation. Do not report stylistic differences as defects unless
they change meaning or usability.

Check that tracked `.mo` files contain the effective translations of current
catalogs; byte differences caused by compiler metadata alone are not stale
translations. Keep review-only work
read-only: Sphinx can compile `.mo` files in place, so use a temporary source
copy or restore only audit-generated changes while preserving pre-existing work.

## Validate and report

Use `.github/workflows/docs.yml` as the build reference. Build EN and RU with
warnings treated as errors where feasible. Separate content warnings from
unavailable external intersphinx inventories. An offline build may disable
`intersphinx_mapping` through the Sphinx Python API without changing repository
configuration. If builds or template generation are blocked, continue with
available source and catalog checks and identify which conclusions remain
unverified. Do not install dependencies or modify configuration outside the
authorized scope just to complete validation.

Check that the expected API objects and their documentation actually appear in
the generated templates and rendered pages. The current Sphinx configuration
mocks optional dependencies and suppresses `autodoc.import_object` warnings, so
an unresolved class can disappear while the build succeeds. Verify affected
objects directly; do not disable or change publication settings just to pass.

When auditing as part of an implementation/update workflow, the initial pass may
identify gaps and defer HTML builds until after corrections. Reuse validation
from the same task only while its code, docstring, RST and catalog inputs remain
unchanged; rerun affected checks after edits. Do not report deferred checks as
passed or repeat a full repository audit for a scoped correction.

A successful build, passing tests and complete catalogs are validation signals,
not proof that all user-visible behavior is documented.

## Reporting and completion

Lead with actionable findings, ordered by user impact. For each finding include:

- Severity: high (likely to break installation, integration or correct use),
  medium (misleading behavior or missing guidance with a practical workaround),
  or low (limited usability impact).
- Confidence: confirmed (direct implementation or reproducible evidence) or
  likely (supporting evidence with a specific unresolved check).
- Type: incorrect claim, missing documentation, translation gap, build defect
  or contract conflict requiring an implementation decision.
- Documentation location and implementation evidence, preferably file and line.
  For missing material, name the expected destination and the relevant code.
- User impact and suggested correction.

Keep unverified areas in a separate coverage summary rather than presenting them
as established defects. Include the baseline or behavior scope, documentation
and catalogs reviewed, EN/RU build results, and skipped or blocked checks.
If no discrepancies were found, say so explicitly within the verified scope.

Finish when each relevant change or behavior in scope has a documentation and
translation status, and each planned validation is completed or explicitly
marked unverified with a reason. Do not expand indefinitely into unrelated
features, or claim exhaustive coverage without tracing all relevant behavior.
For review-only requests, leave corrections as findings. If updates are also
authorized, proceed with those corrections within the requested scope.
