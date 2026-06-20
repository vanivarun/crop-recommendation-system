# Contributing

Thanks for wanting to contribute to the Crop Recommendation System! A few simple steps to get started:

- Fork the repository and create a feature branch: `git checkout -b feat/your-change`
- Keep changes small and focused; open a PR with a clear description and motivation.
- Run tests locally before opening a PR:

```bash
python -m unittest discover -s tests
```

- Run the linter (we use `ruff`):

```bash
pip install -r requirements-dev.txt
ruff check .
```

- If your change touches code formatting or style, run `ruff` fixes where appropriate or follow the repo rules.
- For bug reports, include steps to reproduce, expected vs actual behavior, and environment details.

Maintainers will review and request changes if needed. Thank you!
