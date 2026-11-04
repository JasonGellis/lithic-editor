# Contributing

This page tells you how to set up a development environment and how to submit a change.

## Requirements

- Python 3.10 to 3.13
- Git

## Set up the environment

1. Clone the repository.

    ```bash
    git clone https://github.com/JasonGellis/lithic-editor.git
    cd lithic-editor
    ```

2. Create and activate a virtual environment.

    ```bash
    python -m venv .venv
    source .venv/bin/activate      # Linux and macOS
    .venv\Scripts\activate         # Windows
    ```

3. Install the package in editable mode with the development tools.

    ```bash
    pip install -e ".[dev]"
    ```

    The `dev` extra installs `pytest`, `pytest-qt`, `pytest-cov` and `ruff`.

4. To build the documentation, also install the `docs` extra.

    ```bash
    pip install -e ".[docs]"
    ```

The package depends on `opencv-contrib-python`. The neural upscaling models need the `contrib` build.

## Run the checks

Run these commands before you submit a change.

```bash
ruff check lithic_editor tests
pytest
```

On a Linux machine without a display, set `QT_QPA_PLATFORM=offscreen` before you run `pytest`.

### Lint

The project uses [ruff](https://docs.astral.sh/ruff/). The configuration is in `pyproject.toml` under `[tool.ruff]`:

- Line length: 100.
- Target version: Python 3.10.
- Rules: Pyflakes (`F`) and the pycodestyle error classes `E4`, `E7` and `E9`.
- Ignored rules: `E712` and `F841`.

To apply the safe automatic fixes:

```bash
ruff check --fix lithic_editor tests
```

### Tests

See the [Testing Guide](testing.md) for the test layout, the fixtures and the CI matrix.

### Documentation

```bash
mkdocs serve             # local preview
mkdocs build --strict    # the build that CI runs
```

A GitHub Actions workflow builds the documentation on each push to `main` and deploys it to GitHub Pages.

## Code standards

- Follow [PEP 8](https://peps.python.org/pep-0008/).
- Give each module, class and public function a docstring.
- Give each function one responsibility. If a function does two things, split it.
- Write the documentation and the `--help` text in Simplified Technical English.

### Logging

Library code does not print. Each module in `lithic_editor/processing` logs through `logging.getLogger(__name__)`. The command line sends the `lithic_editor.processing` logger to standard output (`--quiet` keeps warnings only). The window relays it to the log panel through the processing thread.

## Run the application

```bash
lithic-editor --gui
lithic-editor example_images/lithic_300dpi.png --debug
lithic-editor --docs --offline
```

## Project layout

```
lithic_editor/
├── annotations/            # Arrow annotation system
│   ├── arrow_shapes.py     # The geometry of the five arrow styles
│   ├── arrows.py           # Arrow class and the arrow canvas
│   └── integration.py      # The Arrows pane helpers
├── cli/                    # Command-line interface
│   ├── main.py             # Entry point, argument parser and logging set-up
│   ├── help.py             # Help text
│   └── docs_server.py      # Offline documentation server
├── config/                 # config.yaml and its reader
├── gui/
│   ├── main_window.py      # The window: load, process, brush, arrows, save
│   ├── panes.py            # Builds the File, Brush, Arrows and Options panes
│   ├── dialogs.py          # DPI question and upscaling offer
│   ├── brush_canvas.py     # The brush canvas of the Input Image panel
│   ├── debug_panel.py      # The Processing Steps panel
│   ├── processing_thread.py# Runs the pipeline off the window, relays its log
│   └── theme.py            # Colours, stylesheet and icons
├── models/                 # Bundled ESPCN and FSRCNN models (*.pb)
└── processing/
    ├── ripple_removal.py   # process_lithic_drawing: the stages in order
    ├── image_io.py         # Reading the drawing, cropping to content
    ├── preprocess.py       # Smoothing and Sauvola threshold
    ├── cortex.py           # Cortex stipple separation
    ├── skeleton.py         # Thinning, spurs, endpoints and junctions
    ├── segments.py         # Segments and ripple classification
    ├── rebuild.py          # Structural mask and line rebuild
    ├── striking_angle.py   # Striking angle measurement
    ├── debug_images.py     # Debug image writer
    ├── resolution.py       # Line geometry and upscale factor
    └── upscaling.py        # DPI detection and neural upscaling

tests/                      # Test suite (see the Testing Guide)
tools/dpi_eval/             # Evaluation harness
docs/                       # MkDocs source
```

## Extend the evaluation harness

The harness in `tools/dpi_eval/` compares processing strategies across input DPI. The [Testing Guide](testing.md#evaluation-harness) describes how to run it.

### Add a strategy

A strategy is a function in `tools/dpi_eval/strategies.py` with this signature:

```python
def my_strategy(variant_path: Path, dpi: int, workdir: Path) -> np.ndarray:
    """One sentence that says what this strategy changes."""
```

1. Read the input with `load_grayscale(variant_path)`. The file is a grayscale PNG of one DPI variant.
2. Process the image. To call this branch's pipeline, use `_run_pipeline(image, dpi, workdir, **kwargs)`. It writes the console output to `workdir/pipeline.log`.
3. Return the result as a black-on-white `uint8` array. The result can have a different size from the input. The metrics module puts all outputs on a common grid.
4. If the strategy cannot run in this checkout, raise `StrategyUnavailable` with the reason. The harness then skips the strategy and continues.
5. Add the function to the `STRATEGIES` dictionary. The key is the name that `--strategies` accepts.
6. Add a test in `tests/tools/`. `test_dpi_eval_smoke.py` shows the pattern.

### Add a source

A source is a real scan at 600 DPI. The harness makes the 300, 150 and 75 DPI variants from it by area interpolation.

1. Save the scan as `example_images/<name>.png` with a 600 DPI tag.
2. Add `<name>` to `PRIMARY_SOURCES` in `tools/dpi_eval/sources.py`. Without this step, the source runs only when you pass `--sources <name>`.
3. Optional: put a hand-cleaned drawing at `example_images/ground_truth/<name>.png`. The harness then also reports the `gt_*` scores for this source.

To add a real scan to the `--real-scans` set, add a `(<name>, <dpi>)` pair to `REAL_SCANS`. All real scans must show the same drawing.

## Submit a change

1. Create a branch from `develop`.
2. Make the change. Add or update tests for it.
3. Update the documentation if the behaviour changes.
4. Run `ruff check lithic_editor tests` and `pytest`.
5. Push the branch and open a pull request.

The pull request must:

- pass the tests on all CI platforms and Python versions,
- pass `ruff check`,
- include tests for new behaviour,
- describe the change.

## Versions and releases

The project follows [Semantic Versioning](https://semver.org/) from version 1.0.0.

| Change | Version |
|---|---|
| A fix with no change to behaviour | Patch, 1.0.x |
| A new feature or option that keeps existing commands, configuration keys and results | Minor, 1.x.0 |
| A change that breaks the command line, the configuration keys, the CSV columns or the Python API | Major, x.0.0 |

The version number is in one place, `lithic_editor/__init__.py`. To release:

1. Set the version in `lithic_editor/__init__.py` and in `CITATION.cff`. Run
   `python tools/render_banner.py`, because the README banner shows the version.
2. Move the changes in `CHANGELOG.md` under the new version with the date.
3. Merge `develop` into `main`.
4. Tag `main` with `vX.Y.Z` and push the tag.
5. Create the GitHub release from the tag, with the changelog entry as the notes.

## Get help

- Bugs and questions: [GitHub Issues](https://github.com/JasonGellis/lithic-editor/issues)
- Email: jg760@cam.ac.uk
