# Installation

## System Requirements

### Minimum Requirements
- **Python**: 3.7 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 500MB free space

### Operating Systems
- Windows 10/11
- macOS >= 10.15
- Linux (Ubuntu 20.04+, Fedora 34+, etc.)

## Installation Methods

### Method 1: Install from GitHub (Recommended)

This method installs the latest stable version directly from the repository.

```bash
# Install directly from GitHub
pip install git+https://github.com/JasonGellis/lithic-editor.git
```

To install a specific version:

```bash
# Install a specific release
pip install git+https://github.com/JasonGellis/lithic-editor.git@v1.0.0
```

### Method 2: Development Installation

For developers, contributors, or users who want to modify the code:

```bash
# Clone the repository
git clone https://github.com/JasonGellis/lithic-editor.git
cd lithic-editor

# Basic development install (editable mode)
pip install -e .

# Install with test dependencies (recommended for developers)
pip install -e ".[test]"

# Install with all development tools
pip install -e ".[dev]"

# Install with documentation tools
pip install -e ".[docs]"

# Install everything (dev + test + docs)
pip install -e ".[dev,test,docs]"
```

## Dependency Installation


### Dependency Groups Explained

| Group | Purpose | Includes |
|-------|---------|----------|
| *none* | Users only | Core dependencies for running the app |
| `test` | Running tests | pytest, pytest-qt, pytest-cov |
| `dev` | Code quality | black, flake8, mypy + test dependencies |
| `docs` | Building docs | mkdocs, mkdocs-material + extensions (for contributors only) |

All dependencies are automatically installed with the package. However, if you encounter issues, you can manually install them:

```bash
# Core dependencies
pip install numpy opencv-python Pillow PyQt5 scikit-image networkx scipy

# Optional: For building/editing documentation (NOT needed for viewing docs)
# Users can view docs with 'lithic-editor docs' without these packages
pip install mkdocs mkdocs-material pymdown-extensions
```

## Verifying Installation

After installation, verify everything is working:

```bash
# Check version
lithic-editor --version

# Run help command
lithic-editor --help

# Launch GUI (opens a window)
lithic-editor --gui
```

## Accessing Documentation

### ![](../assets/images/web.svg){: style="width:24px; height:24px; vertical-align:text-bottom; filter:brightness(0); margin-right:8px"} Online Documentation
Complete, up to date documentation is always available online:
**https://jasongellis.github.io/lithic-editor/**

### ![](../assets/images/computer.svg){: style="width:24px; height:24px; vertical-align:text-bottom; filter:brightness(0); margin-right:8px"} Local Documentation (Following Installation)
```bash
# Open documentation in your browser
lithic-editor docs

# Serve documentation locally (for offline viewing)
lithic-editor docs --offline
```

### ![](../assets/images/quick_reference.svg){: style="width:24px; height:24px; vertical-align:text-bottom; filter:brightness(0); margin-right:8px"} Quick Help
```bash
lithic-editor --help     # Show all commands and options
lithic-editor help       # Show detailed help information
lithic-editor help api   # Show Python API usage examples
```

### What's in the Documentation?
- **Installation Guide** - Detailed setup instructions (this page!)
- **Quick Start Tutorial** - Process your first image in minutes
- **User Guide** - Complete feature documentation with examples
- **Developer Guide** - Contributing, testing, and extending the application
- **API Reference** - Python API documentation for programmatic usage

## Troubleshooting

### Common Issues

??? failure "ImportError: No module named 'PyQt5'"
    **Solution**: Install PyQt5 manually
    ```bash
    pip install PyQt5==5.15.9
    ```

??? failure "OpenCV import error"
    **Solution**: Reinstall OpenCV
    ```bash
    pip uninstall opencv-python opencv-python-headless
    pip install opencv-python
    ```

??? failure "GUI doesn't launch on Linux"
    **Solution**: Install system dependencies
    ```bash
    # Ubuntu/Debian
    sudo apt-get install python3-pyqt5 libxcb-xinerama0

    # Fedora
    sudo dnf install python3-qt5
    ```

??? failure "Permission denied error"
    **Solution**: Install in user space
    ```bash
    pip install --user git+https://github.com/JasonGellis/lithic-editor.git
    ```

### Virtual Environment Setup

We recommend using a virtual environment:

=== "venv"
    ```bash
    # Create virtual environment
    python -m venv lithic-env

    # Activate it
    # Windows
    lithic-env\Scripts\activate
    # macOS/Linux
    source lithic-env/bin/activate

    # Install package
    pip install git+https://github.com/JasonGellis/lithic-editor.git
    ```

=== "conda"
    ```bash
    # Create conda environment
    conda create -n lithic python=3.9
    conda activate lithic

    # Install package
    pip install git+https://github.com/JasonGellis/lithic-editor.git
    ```

## Platform-Specific Notes

### macOS

On macOS, you might need to allow the application in Security & Privacy settings the first time you run it.

### Windows

Windows users might need to install Visual C++ redistributables if not already present:
- [Download from Microsoft](https://aka.ms/vs/17/release/vc_redist.x64.exe)

### Linux

Some Linux distributions require additional packages:

```bash
# Ubuntu/Debian
sudo apt-get install python3-dev python3-pip python3-venv

# Fedora
sudo dnf install python3-devel

# Arch
sudo pacman -S python python-pip
```

## Updating

To update to the latest version:

```bash
# Update from GitHub
pip install --upgrade git+https://github.com/JasonGellis/lithic-editor.git

# For development installation
cd lithic-editor
git pull
pip install -e . --upgrade
```

## Uninstallation

To remove Lithic Editor:

```bash
pip uninstall lithic-editor
```

## Next Steps

Once installed, proceed to the [Quick Start Guide](quickstart.md) to process your first lithic drawing!