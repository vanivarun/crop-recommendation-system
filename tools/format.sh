#!/usr/bin/env bash
# Run ruff to auto-fix issues
if ! command -v ruff >/dev/null 2>&1; then
  echo "ruff is not installed. Install dev deps: pip install -r requirements-dev.txt"
  exit 1
fi

ruff check --fix .
