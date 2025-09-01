# Requirements

## System Requirements

### Minimum Specifications
- **Processor**: Dual-core CPU (2.0 GHz or faster)
- **Memory**: 4 GB RAM
- **Storage**: 500 MB available space
- **Display**: 1280×720 resolution
- **Graphics**: OpenGL 2.0 support

### Recommended Specifications
- **Processor**: Quad-core CPU (3.0 GHz or faster)
- **Memory**: 8 GB RAM or more
- **Storage**: 2 GB available space
- **Display**: 1920×1080 resolution or higher
- **Graphics**: Dedicated graphics card

## Software Requirements

### Python Version
- **Minimum**: Python 3.7
- **Recommended**: Python 3.9 or later
- **Maximum tested**: Python 3.11

### Operating Systems

=== "Windows"
    - Windows 10 (64-bit) or later
    - Windows 11 fully supported
    - Requires Visual C++ Redistributable

=== "macOS"
    - macOS 10.15 Catalina or later
    - Apple Silicon (M1/M2) supported
    - Intel Macs supported

=== "Linux"
    - Ubuntu 20.04 LTS or later
    - Fedora 34 or later
    - Debian 11 or later
    - Other distributions with Qt5 support

## Python Dependencies

All dependencies are automatically installed with the package:

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥1.20.0 | Numerical operations |
| opencv-contrib-python | ≥4.5.0 | Image processing and neural network upscaling |
| Pillow | ≥8.0.0 | Image I/O |
| PyQt5 | ≥5.15.0 | GUI framework |
| scikit-image | ≥0.18.0 | Advanced image processing |
| networkx | ≥2.5 | Graph algorithms |
| scipy | ≥1.7.0 | Scientific computing |

## Optional Dependencies

### For Development
- pytest (≥6.0) - Testing framework
- black (≥21.0) - Code formatting
- flake8 (≥3.8) - Code linting
- mypy (≥0.800) - Type checking

### For Documentation
- mkdocs (≥1.5.0) - Documentation generator
- mkdocs-material (≥9.0.0) - Material theme
- pymdown-extensions (≥10.0) - Markdown extensions

## Image Format Support

### Input Formats
- **PNG** - Recommended, lossless
- **JPEG/JPG** - Widely supported
- **TIFF/TIF** - Professional quality
- **BMP** - Uncompressed

### Output Formats
- **PNG** - Best for web and presentations
- **JPEG** - Best for publications
- **TIFF** - Best for archival

## Performance Considerations

### Image Size Recommendations
- **Optimal**: 2000-3000 pixels maximum dimension
- **Maximum**: 8000 pixels (may be slow)
- **DPI**: 300 DPI for best quality

### Memory Usage
Memory usage depends on image size:
- 2000×2000 image: ~50 MB
- 4000×4000 image: ~200 MB
- 8000×8000 image: ~800 MB

## Network Requirements

- No internet connection required for core functionality
- Internet needed only for:
  - Installation from GitHub
  - Downloading updates
  - Accessing online documentation

## Compatibility Notes

### Known Compatible IDEs
- Visual Studio Code
- PyCharm
- Jupyter Notebook/Lab
- Spyder
- Sublime Text

### Virtual Environments
Compatible with:
- venv
- virtualenv
- conda/miniconda
- poetry
- pipenv

## Checking Your System

To verify your system meets the requirements:

```python
import sys
import platform

print(f"Python version: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"Architecture: {platform.machine()}")
print(f"Processor: {platform.processor()}")

# Check for required packages
required = ['numpy', 'cv2', 'PIL', 'PyQt5', 'skimage', 'networkx', 'scipy']
for package in required:
    try:
        __import__(package)
        print(f"✓ {package} installed")
    except ImportError:
        print(f"✗ {package} missing")
```

## Troubleshooting Requirements

If you encounter issues:

1. **Update pip**: `pip install --upgrade pip`
2. **Update setuptools**: `pip install --upgrade setuptools wheel`
3. **Check Python version**: `python --version`
4. **Verify 64-bit Python**: `python -c "import sys; print(sys.maxsize > 2**32)"`

For platform-specific issues, see the [Installation Guide](installation.md).