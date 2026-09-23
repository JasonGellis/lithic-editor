# Requirements

## Software Requirements

### Python Version

- **Minimum**: Python 3.10
- **Maximum tested**: Python 3.13

### Operating Systems

The test suite runs on Ubuntu, Windows and macOS.

=== "Windows"
    - Windows 10 or later (64-bit)

=== "macOS"
    - Apple Silicon and Intel Macs

=== "Linux"
    - Ubuntu and other distributions with Qt5 support

## Python Dependencies

pip installs all dependencies with the package.

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥1.24 | Array operations |
| opencv-contrib-python | ≥4.8 | Image processing and the neural upscaling models |
| Pillow | ≥10.0 | Image input and output |
| PyQt5 | ≥5.15.10 | GUI framework |
| scikit-image | ≥0.22 | Thresholding and skeletonization |
| networkx | ≥3.0 | Graph analysis |
| scipy | ≥1.10 | Scientific computing |

!!! note "opencv-contrib-python"
    The neural upscaling models need the `contrib` build of OpenCV. The plain `opencv-python` package does not include them.

## Optional Dependencies

### For Development

- pytest (≥8.0), pytest-qt (≥4.4), pytest-cov (≥5.0) - Tests
- ruff (≥0.6) - Linter

### For Documentation

- mkdocs (≥1.6) - Documentation builder
- mkdocs-material (≥9.5) - Material theme
- mkdocs-material-extensions (≥1.3) - Theme extensions
- pymdown-extensions (≥10.0) - Markdown extensions

## Image Formats

### Input Formats

- **PNG** - Recommended, no loss
- **JPEG/JPG**
- **TIFF/TIF**
- **BMP**

### Output Formats

- **PNG**
- **JPEG**
- **TIFF**

## Image Guidance

- Use a clean black line drawing on white.
- Scan at 300 to 600 DPI.
- Save the scan as PNG.
- Keep the DPI tag in the file. If the file has no tag, the software asks for the DPI.
- Scans at lower resolutions (75 or 150 DPI) are upscaled before processing when you allow it.
- Scan the scale bar as a separate image at the same DPI as the drawing.

## Network Requirements

Processing does not need an internet connection. The neural models are included in the package.

You need an internet connection only to:

- Install or update the package from GitHub
- Open the online documentation

## Virtual Environments

The package installs in venv, virtualenv, conda and other Python environment managers.

## Check Your System

Run this script to check your Python version and the required packages:

```python
import sys
import platform

print(f"Python version: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"Architecture: {platform.machine()}")

# Check the required packages
required = ['numpy', 'cv2', 'PIL', 'PyQt5', 'skimage', 'networkx', 'scipy']
for package in required:
    try:
        __import__(package)
        print(f"✓ {package} installed")
    except ImportError:
        print(f"✗ {package} missing")
```

## Troubleshooting

If the installation fails:

1. Update pip: `pip install --upgrade pip`
2. Update setuptools: `pip install --upgrade setuptools wheel`
3. Check the Python version: `python --version`
4. Check for 64-bit Python: `python -c "import sys; print(sys.maxsize > 2**32)"`

For platform-specific problems, see the [Installation Guide](installation.md).
