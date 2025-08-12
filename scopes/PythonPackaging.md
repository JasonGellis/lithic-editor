# Python Packaging Specification

## Purpose & User Problem

Convert the Lithic Editor and Annotator from a standalone script-based application into a proper Python package that can be:
1. **Installed via git-based pip** for easy distribution in private repositories
2. **Used programmatically** in other archaeological/image processing applications  
3. **Embedded as GUI components** within larger PyQt applications
4. **Maintained as standalone tool** for direct archaeological workflows

## Success Criteria

### Installation & Distribution
- [ ] Package installable via `pip install git+https://github.com/user/lithic-editor.git`
- [ ] CLI help accessible via `lithic-editor --help`
- [ ] Comprehensive help covering usage, API, and GUI workflows
- [ ] Semantic versioning implemented (v1.0.0)

### Programmatic API
- [ ] Clean modular imports: `from lithic_editor.processing import process_lithic_drawing_improved`
- [ ] GUI embeddable: `from lithic_editor.gui import LithicEditorWidget`
- [ ] Arrow system accessible: `from lithic_editor.annotations import Arrow`

### Development Workflow
- [ ] `python lithic_GUI.py` continues to work during development
- [ ] pytest test suite with >80% coverage
- [ ] PEP8 compliant docstrings throughout

### Quality Assurance
- [ ] All current functionality preserved
- [ ] No breaking changes to existing image processing algorithms
- [ ] PyQt5 dependencies properly managed

## Scope & Implementation

### Package Structure
```
lithic-editor/
├── lithic_editor/              # Main package
│   ├── __init__.py            # Package exports
│   ├── __main__.py            # CLI entry point
│   ├── gui/                   # GUI components
│   │   ├── __init__.py
│   │   ├── main_window.py     # Refactored lithic_GUI.py
│   │   ├── canvas.py          # Arrow canvas widget
│   │   └── dialogs.py         # Dialogs and utilities
│   ├── processing/            # Core algorithms
│   │   ├── __init__.py
│   │   ├── ripple_removal.py  # Refactored ripple_remover.py
│   │   └── image_utils.py     # Common image operations
│   ├── annotations/           # Arrow system
│   │   ├── __init__.py
│   │   ├── arrows.py          # Arrow class
│   │   └── integration.py     # Integration helpers
│   └── cli/                   # Command line interface
│       ├── __init__.py
│       └── help.py            # Help system
├── tests/                     # pytest test suite
│   ├── __init__.py
│   ├── test_processing/
│   ├── test_annotations/
│   └── test_gui/
├── docs/                      # Documentation
├── setup.py                   # Package configuration
├── pyproject.toml            # Modern packaging config
├── requirements.txt          # Dependencies
├── requirements-dev.txt      # Development dependencies
├── README.md                 # Updated for package usage
├── LICENSE
└── lithic_GUI.py             # Preserved for development
```

### API Design

#### Processing Module
```python
from lithic_editor.processing import (
    process_lithic_drawing_improved,
    improve_line_quality_antialias
)

# Batch processing in other applications
result = process_lithic_drawing_improved(
    image_path="lithic.png",
    output_folder="results/",
    save_debug=True
)
```

#### GUI Module  
```python
from lithic_editor.gui import LithicEditorWidget, launch_gui

# Standalone launch
launch_gui()

# Embed in other PyQt applications
from PyQt5.QtWidgets import QApplication
app = QApplication([])
editor = LithicEditorWidget()
editor.show()
```

#### Annotations Module
```python
from lithic_editor.annotations import Arrow, ArrowCanvasWidget

# Create arrows programmatically
arrow = Arrow(position=(100, 200), angle=45, size=30)
canvas = ArrowCanvasWidget()
canvas.add_arrow(arrow)
```

### CLI Interface
```bash
# Installation
pip install git+https://github.com/user/lithic-editor.git

# Help system
lithic-editor --help

# Launch GUI
lithic-editor --gui

# Process image via CLI
lithic-editor process input.png --output results/ --debug
```

### Test Coverage Requirements

#### Processing Tests
- [ ] Ripple removal algorithm accuracy
- [ ] DPI preservation throughout pipeline
- [ ] Multiple image format support
- [ ] Error handling for corrupted images

#### GUI Tests  
- [ ] Widget creation and initialization
- [ ] Arrow manipulation interactions
- [ ] File I/O operations
- [ ] Cross-platform compatibility

#### Integration Tests
- [ ] API imports work correctly
- [ ] CLI commands execute properly
- [ ] Package installation succeeds
- [ ] Version management functions

## Technical Considerations

### Dependencies Management
- **PyQt5**: Required dependency (not optional)
- **Core processing**: OpenCV, scikit-image, NetworkX, NumPy, SciPy
- **Testing**: pytest, pytest-qt for GUI testing
- **Development**: black, flake8, mypy for code quality

### Backward Compatibility
- Preserve all existing function signatures
- Maintain current image processing quality
- Keep debug output format consistent
- Ensure DPI handling remains accurate

### Version Management
- Use semantic versioning (MAJOR.MINOR.PATCH)
- Tag releases in git for pip installation
- Maintain CHANGELOG.md for version history
- Setup version string in `__init__.py`

## Out of Scope

### Not Included in This Phase
- [ ] Public PyPI publication (private git repos only)
- [ ] Alternative GUI frameworks (PyQt5 required)
- [ ] Breaking changes to existing algorithms
- [ ] Performance optimizations beyond current functionality
- [ ] Additional image processing features

### Future Considerations
- Documentation website generation
- CI/CD pipeline setup
- Type hints throughout codebase
- Plugin architecture for extensions

## Constraints

### Development Environment
- Must work with existing Python 3.7+ requirement
- Preserve macOS/Windows/Linux compatibility
- Maintain current dependency versions for stability

### Archaeological Requirements
- No changes to ripple removal algorithm accuracy
- DPI preservation critical for publication workflows
- Arrow annotation precision must be maintained
- Debug output format expected by archaeological workflows

## Validation Criteria

### Installation Success
```bash
pip install git+https://repo.git
lithic-editor --help
python -c "from lithic_editor.processing import process_lithic_drawing_improved"
```

### Functionality Preservation
- [ ] Process test images and compare results with current version
- [ ] Verify arrow annotation accuracy and manipulation
- [ ] Confirm DPI preservation through full pipeline
- [ ] Test GUI embedding in sample PyQt application

### Code Quality
- [ ] All tests pass with >80% coverage
- [ ] PEP8 compliance verified
- [ ] No breaking changes to public interfaces
- [ ] Documentation complete and accurate