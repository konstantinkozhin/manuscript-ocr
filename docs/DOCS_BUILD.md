# Docs Build Notes

For maintainers, contributors, and future automation agents working in this
repository.

This file describes the build flow that is known to work locally for the
versioned Sphinx site and the Russian PDF.

## Rules

- Use `python3 -m sphinx`, not bare `sphinx-build`.
- Pass the docs version explicitly through both `release` and `version`.
- Build versioned HTML into `docs/sphinx/_site/<VERSION>/...`.
- Build the Russian PDF through LaTeX output plus `tectonic`.
- Keep this file aligned with [docs.yml](./../.github/workflows/docs.yml) when
  the publishing flow changes.

## Prerequisites

Install the package and the Sphinx toolchain:

```bash
python3 -m pip install -e .
python3 -m pip install sphinx sphinx-rtd-theme numpydoc sphinx-autodoc-typehints sphinxcontrib-mermaid
```

Install `tectonic` for PDF generation.

On macOS:

```bash
brew install tectonic
```

## Build Versioned HTML

```bash
VERSION=0.1.12

mkdir -p docs/sphinx/_site/$VERSION/en docs/sphinx/_site/$VERSION/ru

python3 -m sphinx -b html \
  -D language=en \
  -D release=$VERSION \
  -D version=$VERSION \
  docs/sphinx docs/sphinx/_site/$VERSION/en

python3 -m sphinx -b html \
  -D language=ru \
  -D release=$VERSION \
  -D version=$VERSION \
  docs/sphinx docs/sphinx/_site/$VERSION/ru
```

Result:

- `docs/sphinx/_site/$VERSION/en`
- `docs/sphinx/_site/$VERSION/ru`

## Build Russian PDF

```bash
VERSION=0.1.12

python3 -m sphinx -b latex \
  -D language=ru \
  -D release=$VERSION \
  -D version=$VERSION \
  -D latex_engine=xelatex \
  docs/sphinx docs/sphinx/_build/latex-ru-$VERSION

cd docs/sphinx/_build/latex-ru-$VERSION
tectonic --keep-logs --keep-intermediates manuscript-ocr.tex
```

Result:

- `docs/sphinx/_build/latex-ru-$VERSION/manuscript-ocr.pdf`

Optional convenience copy:

```bash
cp docs/sphinx/_build/latex-ru-$VERSION/manuscript-ocr.pdf \
   docs/sphinx/_build/manuscript-ocr-ru-$VERSION.pdf
```

## Notes

- In this environment Sphinx may print `RuntimeError: _ARRAY_API is not
  PyCapsule object` before continuing. Treat the final exit code and final
  `build succeeded` message as the source of truth.
- PDF generation may warn that `mmdc` is unavailable. The build still
  completes unless Mermaid output is required for a specific page.
- The first `tectonic` run downloads TeX assets from the network, so PDF
  generation is slower on a cold cache.
