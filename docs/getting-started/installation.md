# Installation

## System Requirements

- **Python**: 3.10 to 3.13
- **Operating system**: Windows, macOS or Linux

The test suite runs on Ubuntu, Windows and macOS for Python 3.10 to 3.13.

## Step 1: Prerequisites

Before you install Lithic Editor, make sure that you have:

- **Python 3.10 or later** ([Download Python](https://www.python.org/downloads/))
- **Git**, to install from GitHub ([Download Git](https://git-scm.com/downloads))
- **pip**, the Python package manager (included with Python)

Check your setup:

```bash
python --version    # Shows 3.10 or later
git --version       # Shows the git version
pip --version       # Shows the pip version
```

## Step 2: Virtual Environment (Recommended)

Create a separate environment for Lithic Editor.

=== "venv"
    ```bash
    # Create the virtual environment
    python -m venv lithic-env

    # Activate it
    # Windows
    lithic-env\Scripts\activate
    # macOS/Linux
    source lithic-env/bin/activate
    ```

=== "conda"
    ```bash
    # Create the conda environment
    conda create -n lithic python=3.12
    conda activate lithic
    ```

## Step 3: Installation

Choose one installation method.

### Method 1: Install from GitHub (Most Users)

Activate your virtual environment. Then run:

```bash
# Install from GitHub
pip install git+https://github.com/JasonGellis/lithic-editor.git
```

To install a tagged release, add `@<tag>` to the end of the URL.

### Method 2: Development Installation (Contributors)

Use this method if you want to change the code.

```bash
# Clone the repository
git clone https://github.com/JasonGellis/lithic-editor.git
cd lithic-editor

# Editable install
pip install -e .

# Editable install with the test tools
pip install -e ".[test]"

# Editable install with the test tools and ruff
pip install -e ".[dev]"

# Editable install with the documentation tools
pip install -e ".[docs]"

# Editable install with all tools
pip install -e ".[dev,docs]"
```

## Step 4: Check the Installation

Run these commands:

```bash
# Show the version
lithic-editor --version

# Show the help
lithic-editor --help

# Start the GUI (opens a window)
lithic-editor --gui
```

## Dependencies

### Core Dependencies

pip installs all core dependencies with the package.

| Package | Purpose |
|---------|---------|
| numpy | Array operations |
| opencv-contrib-python | Image processing and the neural upscaling models |
| Pillow | Image input and output |
| PyQt5 | GUI framework |
| scikit-image | Thresholding and skeletonization |
| networkx | Graph analysis |
| scipy | Scientific computing |

!!! note "opencv-contrib-python"
    The neural upscaling models need the `contrib` build of OpenCV. The plain `opencv-python` package does not include them.

### Optional Dependencies

| Group | Purpose | Packages |
|-------|---------|----------|
| `test` | Run the tests | pytest, pytest-qt, pytest-cov |
| `dev` | Run the tests and the linter | `test` packages and ruff |
| `docs` | Build the documentation | mkdocs, mkdocs-material, mkdocs-material-extensions, pymdown-extensions |

If the automatic installation fails, install the core dependencies by hand:

```bash
# Core dependencies
pip install numpy opencv-contrib-python Pillow PyQt5 scikit-image networkx scipy
```

You do not need the `docs` packages to read the documentation. Run `lithic-editor docs` to open the online documentation.

## Troubleshooting

### Common Problems

??? failure "ImportError: No module named 'PyQt5'"
    **Solution**: Install PyQt5 by hand.
    ```bash
    pip install PyQt5
    ```

??? failure "OpenCV import error"
    **Solution**: Remove all OpenCV packages. Then install `opencv-contrib-python`.
    ```bash
    pip uninstall opencv-python opencv-python-headless opencv-contrib-python
    pip install opencv-contrib-python
    ```

??? failure "The GUI does not start on Linux"
    **Solution**: Install the system packages for Qt.
    ```bash
    # Ubuntu/Debian
    sudo apt-get install python3-pyqt5 libxcb-xinerama0

    # Fedora
    sudo dnf install python3-qt5
    ```

??? failure "Permission denied"
    **Solution**: Install in your user directory.
    ```bash
    pip install --user git+https://github.com/JasonGellis/lithic-editor.git
    ```

### Platform Notes

=== "Windows"
    Some systems need the Visual C++ redistributable:
    - [Download from Microsoft](https://aka.ms/vs/17/release/vc_redist.x64.exe)

=== "macOS"
    The first time you start the GUI, macOS can ask for permission in Security & Privacy. Give the permission.

=== "Linux"
    Some distributions need more packages:
    ```bash
    # Ubuntu/Debian
    sudo apt-get install python3-dev python3-pip python3-venv

    # Fedora
    sudo dnf install python3-devel

    # Arch
    sudo pacman -S python python-pip
    ```

## Update

To update to the latest version:

```bash
# Update from GitHub
pip install --upgrade git+https://github.com/JasonGellis/lithic-editor.git

# Update a development installation
cd lithic-editor
git pull
pip install -e . --upgrade
```

## Uninstall

To remove Lithic Editor:

```bash
pip uninstall lithic-editor
```

## Next Steps

Go to the [User Guide](../user-guide/overview.md) to process your first drawing.
