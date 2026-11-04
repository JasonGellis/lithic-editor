# Contributing

Thank you for your interest in Lithic Editor. Contributions of every kind are welcome:
bug reports, feature requests, documentation, test drawings and code.

## Report a bug or request a feature

Open an issue at <https://github.com/JasonGellis/lithic-editor/issues>. Use the issue
templates. For a bug, give the version (`lithic-editor --version`), the operating system,
the exact command or the steps in the GUI, and the drawing if you can share it.

## Change the code or the documentation

The full guide is in the documentation:
<https://jasongellis.github.io/lithic-editor/developer/contributing/>. In short:

1. Fork the repository and create a branch from `develop`.
2. Install the development dependencies: `pip install -e ".[dev,docs]"`.
3. Make the change. Add or update the tests.
4. Run the checks: `ruff check lithic_editor tests`, `pytest`, `mkdocs build --strict`.
5. Write user-facing text in Simplified Technical English, as the rest of the project does.
6. Open a pull request against `develop`. Describe the change and the reason for it.

## Versions and releases

The project follows [Semantic Versioning](https://semver.org/) from version 1.0.0:

- **Patch** (1.0.x): a fix with no change to behaviour.
- **Minor** (1.x.0): a new feature or option that keeps existing commands, configuration
  keys and results working.
- **Major** (x.0.0): a change that breaks the command line, the configuration keys, the
  CSV columns or the Python API.

The version number lives in one place, `lithic_editor/__init__.py`. A release is a merge
of `develop` into `main`, a `vX.Y.Z` tag on `main`, an entry in `CHANGELOG.md` and a
GitHub release with the same notes.

## Code of conduct

Everyone who takes part in the project follows the [code of conduct](CODE_OF_CONDUCT.md).

## Questions

Open an issue, or write to the maintainer at <jg760@cam.ac.uk>.
