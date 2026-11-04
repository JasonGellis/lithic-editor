# Changelog

All notable changes to Lithic Editor are recorded in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The version
numbers follow [Semantic Versioning](https://semver.org/): a patch version for a fix, a minor
version for a new feature, a major version for a change that breaks the command line, the
configuration keys or the Python API. Each release has a `vX.Y.Z` tag and a GitHub release
with these notes.

## [1.0.0] - Unreleased

The first release.

### Added
- Ripple removal for scanned lithic drawings. The pipeline keeps the scar contours, the
  outline and the cortex stipple.
- Content-adaptive upscaling for low-resolution scans with the ESPCN and FSRCNN models, and
  a restore step that returns the result at the input size.
- Striking angle measurement: the direction of the blow for each scar from the arc of its
  ripples, in the angle frame PyLithics uses. The GUI adds an arrow for each scar. The CLI
  writes a CSV file. Thresholds: `--min-ripples`, `--min-arc-depth`, `--max-cortex-dots`.
- Arrow annotation in the GUI: add, move, rotate, resize, colour and delete arrows, in five
  styles from the conventions of lithic illustration (Filled, Open, Dot, Crossed, Dashed).
- A brush to paint on the result before arrows are added.
- A command line with one flag for each action: `--gui`, `--docs`, `--help`, `--help-api`,
  `--version`, and a positional drawing with processing options.
- Configuration in `config.yaml`, with `--config` for a whole file.
- Debug images for each processing step, and a Processing Steps panel in the GUI.
- Documentation site with user guide, CLI reference, Python API and glossary.
- A DPI evaluation harness in `tools/dpi_eval`.

### Changed
- The pipeline reports through Python logging. The command line prints the log lines
  unless `--quiet` is given, which keeps warnings only. The window shows them in its log.
- NetworkX is no longer a dependency. The pipeline built a graph it never read.

[1.0.0]: https://github.com/JasonGellis/lithic-editor/releases/tag/v1.0.0
