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

## Step 1: Prerequisites

Before installing Lithic Editor, ensure you have:

- **Python 3.7+** installed on your system ([Download Python](https://www.python.org/downloads/))
- **Git** installed for GitHub access ([Download Git](https://git-scm.com/downloads))
- **pip** package manager (included with Python 3.4+)

Verify your setup:
```bash
python --version    # Should show 3.7 or higher
git --version       # Should show git version
pip --version       # Should show pip version
```

## Step 2: Virtual Environment Setup (Recommended)

Create an isolated environment for Lithic Editor:

=== "venv"
    ```bash
    # Create virtual environment
    python -m venv lithic-env

    # Activate it
    # Windows
    lithic-env\Scripts\activate
    # macOS/Linux
    source lithic-env/bin/activate
    ```

=== "conda"
    ```bash
    # Create conda environment
    conda create -n lithic python=3.9
    conda activate lithic
    ```

## Step 3: Installation

Choose your installation method:

### Method 1: Install from GitHub (Most Users)

With your virtual environment activated:

```bash
# Install directly from GitHub
pip install git+https://github.com/JasonGellis/lithic-editor.git
```

To install a specific version:
```bash
# Install a specific release
pip install git+https://github.com/JasonGellis/lithic-editor.git@v1.0.0
```

### Method 2: Development Installation (Contributors)

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

## Step 4: Verify Installation

Test that everything is working:

```bash
# Check version
lithic-editor --version

# Run help command
lithic-editor --help

# Launch GUI (opens a window)
lithic-editor --gui
```

## Dependency Information

### Core Dependencies
All dependencies are automatically installed:

| Package | Purpose |
|---------|---------|
| numpy | Numerical operations |
| opencv-python | Image processing |
| Pillow | Image I/O |
| PyQt5 | GUI framework |
| scikit-image | Advanced image processing |
| networkx | Graph algorithms |
| scipy | Scientific computing |

### Development Dependencies

| Group | Purpose | Includes |
|-------|---------|----------|
| `test` | Running tests | pytest, pytest-qt, pytest-cov |
| `dev` | Code quality | black, flake8, mypy + test dependencies |
| `docs` | Building docs | mkdocs, mkdocs-material + extensions |

All dependencies are automatically installed with the package. However, if you encounter issues, you can manually install them:

```bash
# Core dependencies
pip install numpy opencv-python Pillow PyQt5 scikit-image networkx scipy

# Optional: For building/editing documentation (NOT needed for viewing docs)
# Users can view docs with 'lithic-editor docs' without these packages
pip install mkdocs mkdocs-material pymdown-extensions
```

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

### Platform-Specific Setup

=== "Windows"
    You might need Visual C++ redistributables:
    - [Download from Microsoft](https://aka.ms/vs/17/release/vc_redist.x64.exe)

=== "macOS"
    You might need to allow the application in Security & Privacy settings the first time you run it.

=== "Linux"
    Some distributions require additional packages:
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

Once installed, proceed to the [User Guide](../user-guide/overview.md) to begin processing lithic drawings.