# Testing Guide

## Overview

Lithic Editor uses a comprehensive test suite to ensure reliability and maintain code quality. This guide covers how to run, write, and understand the tests.

## Quick Start

### Installation

```bash
# Install with test dependencies
pip install -e ".[test]"

# Or install all development dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_processing.py

# Run with coverage
pytest --cov=lithic_editor --cov-report=html
```

## Test Structure

### Test Organization

```
tests/
├── conftest.py           # Shared fixtures and configuration
├── test_processing.py    # Image processing tests
├── test_annotations.py   # Arrow annotation tests  
├── test_cli.py          # Command-line interface tests
├── test_gui.py          # GUI component tests
└── README.md            # Test documentation
```

### Test Categories

#### 1. Processing Tests (`test_processing.py`)

Tests the core image processing algorithms:

- Image loading from files and numpy arrays
- Ripple removal algorithm correctness
- Debug output generation
- DPI preservation and handling
- Multiple image format support
- Error handling and edge cases

**Example:**
```python
def test_process_image_from_file(sample_image, temp_dir):
    """Test processing an image from file path."""
    result = process_lithic_drawing(
        image_path=str(sample_image),
        output_folder=str(temp_dir),
        save_debug=False
    )
    
    assert result is not None
    assert isinstance(result, np.ndarray)
    assert result.shape == (100, 100)
```

#### 2. Annotation Tests (`test_annotations.py`)

Tests the arrow annotation system:

- Arrow creation and customization
- Detection status calculations  
- Canvas widget functionality
- Coordinate mapping between view and image
- DPI-aware arrow sizing

**Example:**
```python
def test_arrow_detection_status():
    """Test arrow detection status calculation."""
    small_arrow = Arrow(size=10)
    assert small_arrow.get_detection_status() == "error"
    
    large_arrow = Arrow(size=100)
    assert large_arrow.get_detection_status() == "good"
```

#### 3. CLI Tests (`test_cli.py`)

Tests the command-line interface:

- Argument parsing for all commands
- File validation and error handling
- Help system functionality
- Documentation commands
- Process workflow integration

**Example:**
```python
def test_process_command():
    """Test process command parsing."""
    parser = create_parser()
    args = parser.parse_args(['process', 'image.png', '--debug'])
    
    assert args.command == 'process'
    assert args.input_image == 'image.png'
    assert args.debug == True
```

#### 4. GUI Tests (`test_gui.py`)

Tests the graphical interface:

- Window creation and initialization
- Widget existence and configuration
- Button states and interactions
- Processing thread functionality
- Canvas operations

**Example:**
```python
def test_gui_creation(qapp):
    """Test creating main GUI window."""
    window = LithicProcessorGUI()
    assert window is not None
    assert window.windowTitle() == "Lithic Editor and Annotator"
```

## Test Fixtures

### Available Fixtures

Defined in `conftest.py`:

#### `qapp`
PyQt5 QApplication instance for GUI tests.

#### `temp_dir`
Temporary directory that's automatically cleaned up.

#### `sample_image`
100x100 pixel test image with horizontal lines and a vertical line.

#### `sample_image_with_dpi`  
300x300 pixel test image with DPI metadata.

#### `sample_numpy_array`
NumPy array representing a test image.

#### `sample_pixmap`
QPixmap for GUI testing.

### Using Fixtures

```python
def test_with_fixtures(sample_image, temp_dir):
    """Example test using fixtures."""
    # sample_image is automatically created
    # temp_dir is automatically created and cleaned up
    
    result = process_image(str(sample_image))
    
    output_file = temp_dir / "result.png"
    save_image(result, output_file)
    
    assert output_file.exists()
```

## Writing New Tests

### Test Naming

- File names: `test_*.py`
- Class names: `TestClassName`
- Function names: `test_function_name`
- Use descriptive names that explain what's being tested

### Test Structure

```python
class TestFeature:
    """Test suite for Feature."""
    
    def test_basic_functionality(self):
        """Test basic feature works."""
        # Arrange
        feature = Feature()
        
        # Act
        result = feature.do_something()
        
        # Assert
        assert result is not None
    
    def test_edge_case(self):
        """Test feature handles edge case."""
        # Test implementation
        pass
    
    def test_error_handling(self):
        """Test feature handles errors gracefully."""
        with pytest.raises(ValueError):
            Feature().invalid_operation()
```

### Best Practices

1. **One concept per test**: Each test should focus on one specific behavior
2. **Descriptive names**: Test names should clearly indicate what they test
3. **Use fixtures**: Reuse common setup through fixtures
4. **Mock external dependencies**: Use `unittest.mock` for external services
5. **Test both success and failure**: Include error cases and edge conditions
6. **Keep tests fast**: Avoid unnecessary file I/O or network calls

## Continuous Integration

### GitHub Actions

Tests run automatically on:

- Every push to main branch
- Every pull request
- Multiple platforms: Windows, macOS, Linux
- Multiple Python versions: 3.7-3.11

### Configuration

See `.github/workflows/tests.yml` for the complete CI configuration.

### Special Considerations

**Linux CI (headless)**:
```bash
export QT_QPA_PLATFORM=offscreen
xvfb-run -a pytest
```

**Cross-platform testing**:
- Uses virtual display on Linux
- Handles platform-specific PyQt5 differences
- Tests file path handling across platforms

## Coverage Goals

### Current Coverage

Run with coverage to see current status:
```bash
pytest --cov=lithic_editor --cov-report=html
open htmlcov/index.html
```

### Target Coverage

- **Overall**: 80%+
- **Processing module**: 90%+
- **CLI module**: 85%+
- **GUI module**: 70%+ (some interactions hard to test)

### Improving Coverage

Focus areas for improvement:
- Error handling paths
- Edge cases in image processing
- GUI interaction flows
- File I/O error conditions

## Debugging Tests

### Running Specific Tests

```bash
# Run one test
pytest tests/test_processing.py::TestProcessingModule::test_specific_case -v

# Run tests matching pattern
pytest -k "test_arrow" -v

# Stop on first failure
pytest -x

# Show local variables on failure
pytest --tb=long
```

### Test Debugging

```python
def test_debug_example():
    """Example test with debugging."""
    import pdb; pdb.set_trace()  # Debugger breakpoint
    
    result = function_to_test()
    
    # Or use print statements
    print(f"Result: {result}")
    
    assert result == expected
```

### Common Issues

**PyQt5 issues**:
- Ensure QApplication exists (use `qapp` fixture)
- Set `QT_QPA_PLATFORM=offscreen` for headless testing

**File permissions**:
- Use `temp_dir` fixture for file operations
- Clean up resources in teardown

**Import errors**:
- Install package in development mode: `pip install -e .`
- Check Python path includes project directory

## Performance Testing

### Benchmarking

```python
import time

def test_processing_performance(sample_image):
    """Test processing performance."""
    start_time = time.time()
    
    result = process_lithic_drawing(str(sample_image))
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Should process in reasonable time
    assert processing_time < 30.0  # 30 seconds max
    assert result is not None
```

### Memory Testing

```python
import psutil
import os

def test_memory_usage(large_sample_image):
    """Test memory usage stays reasonable."""
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    
    result = process_lithic_drawing(str(large_sample_image))
    
    final_memory = process.memory_info().rss
    memory_increase = final_memory - initial_memory
    
    # Should not use excessive memory (100MB max increase)
    assert memory_increase < 100 * 1024 * 1024
```

## Contributing Tests

### Before Submitting

1. **Run full test suite**: `pytest`
2. **Check coverage**: `pytest --cov=lithic_editor`
3. **Verify style**: `black tests/` and `flake8 tests/`
4. **Test on your platform**: Ensure tests pass locally

### Pull Request Requirements

- All existing tests must pass
- New features must include tests
- Test coverage should not decrease
- Follow existing test patterns and naming conventions

## Getting Help

- **Test Issues**: [GitHub Issues](https://github.com/JasonGellis/lithic-editor/issues)
- **Test Discussions**: [GitHub Discussions](https://github.com/JasonGellis/lithic-editor/discussions)
- **Documentation**: This guide and test docstrings