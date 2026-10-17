# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Read [AGENTS.md](AGENTS.md)**: it is the source of truth for setup, commands, architecture, conventions, the
annotated test corpus, CI and releasing, and for using the tool from an agent (`--format json|sarif`, the
`ml_code_smell_detector.api` module, rule ids and `# mlscent: ignore[...]` suppressions).

Quick reference:

```bash
uv sync --extra dev                                   # install (dev extras)
uv run python -m pytest tests/                        # all tests (unit, corpus, CLI end-to-end)
uv run ruff check . && uv run python -m flake8 ml_code_smell_detector tests
uv run ml_smell_detector analyze <path> --format json # machine-readable report on stdout
```
