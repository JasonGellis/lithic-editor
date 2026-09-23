# Test Suite

Tests for the Lithic Editor and Annotator. The full description is in `docs/developer/testing.md`.

## Run

```bash
pip install -e ".[test]"
pytest
```

On a Linux machine without a display, set `QT_QPA_PLATFORM=offscreen` first.

```bash
pytest tests/test_processing.py          # one file
pytest tests/test_cli.py::TestCLIParser  # one class
pytest -k "arrow"                        # tests whose name contains "arrow"
pytest --cov=lithic_editor --cov-report=html   # coverage report in htmlcov/
python tests/run_tests.py                # pytest with coverage when pytest-cov is installed
```

## Layout

| File | Covers |
|---|---|
| `conftest.py` | Shared fixtures: `qapp`, `temp_dir`, `sample_image`, `sample_image_with_dpi`, `sample_numpy_array`, `mock_processed_image`, `sample_pixmap`. |
| `test_processing.py` | The processing pipeline: file and array input, debug images, DPI tag, cortex preservation, image formats. |
| `test_resolution.py` | Line geometry measurement, upscale factor choice, and restore to the input grid. |
| `test_upscaling.py` | DPI detection, upscale calculations, model loading, upscaling with fallback, validation. |
| `test_annotations.py` | The `Arrow` class and the arrow canvas. |
| `test_cli.py` | Argument parsing, input validation, the `process`, `help` and `docs` commands, `--keep-upscaled`. |
| `test_docs_server.py` | The offline documentation server. |
| `test_gui.py` | Window creation, widgets, the processing thread, the canvas, and the dialogs. |
| `test_integration.py` | Upscaling with the pipeline, cortex with upscaling, CLI and GUI to processing, debug images, parameter combinations. |
| `tools/test_dpi_eval_metrics.py` | The evaluation harness metrics. |
| `tools/test_dpi_eval_smoke.py` | The `native` strategy of the harness, end to end. |

## Continuous integration

`.github/workflows/tests.yml` runs the suite on Ubuntu, Windows and macOS with Python 3.10, 3.11, 3.12 and 3.13. A separate job runs `ruff check lithic_editor tests`.

## Write a test

- Name files `test_*.py`, classes `Test*`, and functions `test_*`.
- Test one behaviour in each function.
- Use the fixtures in `conftest.py` for images and temporary directories.
- Use the `qapp` fixture in each test that creates a widget.
- Mock file dialogs and other external calls with `unittest.mock`.
