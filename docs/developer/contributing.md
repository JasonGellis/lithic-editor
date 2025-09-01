# Contributing to Lithic Editor

Thank you for your interest in contributing to the Lithic Editor and Annotator! This guide will help you get started.

## Development Setup

### Prerequisites

- Python 3.7 or higher
- Git
- Basic familiarity with PyQt5 and image processing concepts

### Setting Up Your Environment

1. **Fork and clone the repository**:
```bash
git clone https://github.com/YourUsername/lithic-editor.git
cd lithic-editor
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv lithic-env
source lithic-env/bin/activate  # Linux/macOS
# or
lithic-env\Scripts\activate     # Windows
```

3. **Install in development mode**:
```bash
# Install with test dependencies
pip install -e ".[test]"

# Or install with all development tools
pip install -e ".[dev]"

# Or install everything (dev + docs + test)
pip install -e ".[dev,docs,test]"
```

## Development Workflow

### Running Tests

Always run tests before submitting changes:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=lithic_editor --cov-report=html

# Run specific test file
pytest tests/test_processing.py

# Run specific test
pytest tests/test_processing.py::TestProcessingModule::test_process_image_from_file

# View coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

### Code Quality

We use several tools to maintain code quality:

```bash
# Format code
black lithic_editor tests

# Check style
flake8 lithic_editor tests

# Type checking (optional)
mypy lithic_editor
```

### Running the Application

Test your changes:

```bash
# GUI mode
lithic-editor --gui

# CLI mode
lithic-editor process example.png --debug

# Help system
lithic-editor docs --offline
```

## Project Structure

```
lithic_editor/
├── annotations/         # Arrow annotation system
│   ├── arrows.py       # Arrow classes and canvas
│   └── integration.py  # GUI integration helpers
├── cli/                # Command-line interface
│   ├── main.py        # Main CLI entry point
│   ├── help.py        # Help system
│   └── docs_server.py # Documentation server
├── gui/                # Graphical user interface
│   └── main_window.py # Main application window
└── processing/         # Image processing algorithms
    └── ripple_removal.py # Core processing engine

tests/                  # Test suite
├── conftest.py        # Test configuration
├── test_processing.py # Processing tests
├── test_annotations.py # Annotation tests
├── test_cli.py        # CLI tests
└── test_gui.py        # GUI tests

docs/                   # Documentation source
├── index.md           # Homepage
├── user-guide/        # User documentation
├── developer/         # Developer documentation
└── getting-started/   # Installation guides
```

## Writing Tests

### Test Categories

1. **Unit Tests**: Test individual functions/classes
2. **Integration Tests**: Test component interactions
3. **GUI Tests**: Test user interface components
4. **CLI Tests**: Test command-line interface

### Test Guidelines

- Write tests for new features
- Maintain test coverage above 80%
- Use descriptive test names
- Use fixtures for common setup
- Mock external dependencies

### Example Test

```python
def test_process_image_with_debug(sample_image, temp_dir):
    """Test processing with debug output enabled."""
    result = process_lithic_drawing(
        image_path=str(sample_image),
        output_folder=str(temp_dir),
        save_debug=True
    )

    assert result is not None
    debug_files = list(temp_dir.glob("*.png"))
    assert len(debug_files) > 0
```

## Documentation

### Building Documentation

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Serve documentation locally
mkdocs serve

# Build static documentation
mkdocs build

# Deploy to GitHub Pages (maintainers only)
mkdocs gh-deploy
```

### Writing Documentation

- Use clear, concise language
- Include code examples
- Add screenshots for UI features
- Follow the existing structure

## Submission Guidelines

### Pull Request Process

1. **Create a feature branch**:
```bash
git checkout -b feature/your-feature-name
```

2. **Make your changes** following the coding standards

3. **Write tests** for new functionality

4. **Update documentation** if needed

5. **Run the full test suite**:
```bash
pytest
```

6. **Check code quality**:
```bash
black lithic_editor tests
flake8 lithic_editor tests
```

7. **Commit your changes**:
```bash
git add .
git commit -m "Add feature: your feature description"
```

8. **Push and create pull request**:
```bash
git push origin feature/your-feature-name
```

### Pull Request Requirements

- [ ] All tests pass
- [ ] Code coverage maintained or improved
- [ ] Documentation updated (if applicable)
- [ ] Code follows style guidelines
- [ ] Descriptive commit messages
- [ ] Pull request description explains changes

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/JasonGellis/lithic-editor/issues)
- **Discussions**: [GitHub Discussions](https://github.com/JasonGellis/lithic-editor/discussions)
- **Email**: jg760@cam.ac.uk

## Code of Conduct

Please note that this project is released with a Contributor Code of Conduct. By participating in this project you agree to abide by its terms.

## Areas for Contribution

- **Algorithm improvements**: Enhance ripple detection accuracy
- **GUI enhancements**: Improve user experience
- **Documentation**: Add examples and tutorials
- **Testing**: Increase test coverage
- **Performance**: Optimize processing speed
- **Platform support**: Improve cross-platform compatibility

Thank you for contributing! 🏛️