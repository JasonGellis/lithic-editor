# Lithic Editor Test Suite

## Overview

This directory contains the comprehensive test suite for the Lithic Editor and Annotator application.

## Test Structure

```
tests/
├── conftest.py           # Pytest configuration and shared fixtures
├── test_processing.py    # Tests for image processing algorithms
├── test_annotations.py   # Tests for arrow annotation system
├── test_cli.py          # Tests for command-line interface
├── test_gui.py          # Tests for GUI components
└── run_tests.py         # Convenience test runner script
```

## Running Tests

### Quick Start

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_processing.py

# Run specific test class
pytest tests/test_processing.py::TestProcessingModule

# Run specific test
pytest tests/test_processing.py::TestProcessingModule::test_process_image_from_file
```

### With Coverage

```bash
# Install coverage dependencies
pip install pytest-cov

# Run tests with coverage
pytest --cov=lithic_editor --cov-report=html

# View coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

### Using the Test Runner

```bash
# Run the convenience script
python tests/run_tests.py
```

## Test Categories

### Unit Tests

- **Processing Module** (`test_processing.py`)
  - Image loading and format support
  - Ripple removal algorithm
  - Debug output generation
  - DPI preservation
  - Error handling

- **Annotations Module** (`test_annotations.py`)
  - Arrow creation and properties
  - Detection status calculations
  - Canvas widget functionality
  - Coordinate mapping
  - DPI-aware sizing

### Integration Tests

- **CLI Tests** (`test_cli.py`)
  - Argument parsing
  - Command dispatch
  - File validation
  - Process workflow
  - Documentation commands

- **GUI Tests** (`test_gui.py`)
  - Window creation
  - Widget initialization
  - Button states
  - Processing thread
  - Canvas operations

## Test Fixtures

Common fixtures are defined in `conftest.py`:

- `qapp`: PyQt5 QApplication instance
- `temp_dir`: Temporary directory for test files
- `sample_image`: Basic test image
- `sample_image_with_dpi`: Test image with DPI metadata
- `sample_numpy_array`: NumPy array test data
- `sample_pixmap`: QPixmap for GUI tests

## Writing New Tests

### Test File Template

```python
"""
Tests for [module name].
"""

import pytest
from lithic_editor.module import Component

class TestComponent:
    """Test suite for Component."""
    
    def test_basic_functionality(self):
        """Test basic component functionality."""
        component = Component()
        assert component is not None
    
    def test_with_fixture(self, sample_image):
        """Test using fixture."""
        result = process(sample_image)
        assert result is not None
```

### Best Practices

1. **Use descriptive names**: Test names should clearly indicate what they test
2. **One assertion focus**: Each test should focus on one specific behavior
3. **Use fixtures**: Reuse common setup through fixtures
4. **Mock external dependencies**: Use `unittest.mock` for external services
5. **Test edge cases**: Include tests for error conditions and boundaries

## Continuous Integration

Tests are automatically run on:
- Every push to main branch
- Every pull request
- Can be run manually via GitHub Actions

## Coverage Goals

Target coverage: **80%+**

Current focus areas for improvement:
- GUI interaction tests
- File I/O edge cases
- Error recovery paths

## Troubleshooting

### Common Issues

**PyQt5 import errors**
```bash
pip install PyQt5
```

**No display available (Linux CI)**
```bash
export QT_QPA_PLATFORM=offscreen
pytest
```

**Slow test execution**
```bash
# Run tests in parallel
pip install pytest-xdist
pytest -n auto
```

## Contributing

When adding new features:
1. Write tests first (TDD approach)
2. Ensure all tests pass
3. Maintain or improve coverage
4. Update this README if adding new test categories