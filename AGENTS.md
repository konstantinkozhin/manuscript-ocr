# Repository instructions

These instructions apply throughout the repository. Nested `AGENTS.md` files
may provide more specific instructions for their directories.

## Code and compatibility

- Source code lives in `src/manuscript`; tests live in `tests`.
- Preserve the Python versions declared in `pyproject.toml`, including Python
  3.9. Keep dependencies for optional features optional.
- Preserve ordinary inference without PyTorch or training dependencies and
  base-package imports without ONNX Runtime installed.
- Check runtime capabilities explicitly. A CPU/GPU package does not guarantee
  CUDA, DLPack or other optional capabilities.
- Do not hide active inference errors by retrying on another device unless
  the API explicitly defines that behavior.
- Preserve existing user changes. Do not modify `env/` as source code.

## Tests

- Reproduce relevant failures and run tests appropriate to the change. Add
  regression coverage for substantial behavior when needed.
- Keep new tests minimal: reuse fixtures and shared contract coverage; avoid
  redundant cases and broad parameter matrices without a demonstrated need.
- Use `env/bin/python` when available, for example:
  `env/bin/python -m pytest tests/recognizers/test_trocr_device_decoding.py -q`.
- Unit tests for capability selection should explicitly model supported and
  unsupported environments rather than depend on the installed runtime.
- Integration tests may skip unavailable optional dependencies, devices or
  capabilities with a clear reason. Do not skip failures in supported setups.
- Run `git diff --check`. Report checks, skips and limitations accurately;
  do not describe a local test run as a complete CI run.

## Documentation

Keep user-facing documentation and Russian translations consistent with code
changes. Detailed workflows are provided as repository skills:

- [Audit documentation freshness](.agents/skills/docs-audit/SKILL.md)
- [Update documentation and translations](.agents/skills/docs-update/SKILL.md)

Repository skills live in `.agents/skills/` for automatic Codex discovery.

## New processor classes

For new recognizers, detectors, correctors and layouts, follow
[Add a processor class](.agents/skills/add-processor/SKILL.md), including its final
documentation audit and update. Do not integrate new classes into demos or
application defaults unless explicitly requested.
