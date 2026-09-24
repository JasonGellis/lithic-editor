# Installation Guide

This guide gives the procedure to install Lithic Editor. Lithic Editor
operates on macOS, Windows and Linux. Python 3.10 or later is necessary.

## What is necessary

- **Python**: 3.10 to 3.13
- **Operating system**:
  - macOS on Apple Silicon or Intel
  - Windows 10 or later (64-bit)
  - Linux (Ubuntu 22.04 or later, or equivalent, with Qt5 support)
- **Git**: to get the code from GitHub

The neural upscaling models are included in the package. No download is
necessary.

## Step 1: Make sure that Python and Git are installed

### Python

=== "macOS & Linux"

    ```bash
    # Show the Python version. It must be 3.10 or later.
    python3 --version

    # If Python is not installed:
    # macOS (with Homebrew, from https://brew.sh/)
    brew install python@3.12

    # Ubuntu/Debian
    sudo apt-get update
    sudo apt-get install python3 python3-pip python3-venv

    # Fedora
    sudo dnf install python3 python3-pip
    ```

=== "Windows"

    ```powershell
    # Show the Python version. It must be 3.10 or later.
    python --version

    # If Python is not installed, get it from https://python.org
    # During the installation, select "Add python.exe to PATH".
    ```

### Git

=== "macOS & Linux"

    ```bash
    # Show the Git version.
    git --version

    # If Git is not installed:
    # macOS (with Homebrew, from https://brew.sh/)
    brew install git

    # Ubuntu/Debian
    sudo apt-get install git

    # Fedora
    sudo dnf install git
    ```

=== "Windows"

    ```powershell
    # Show the Git version.
    git --version

    # If Git is not installed, get it from https://git-scm.com/
    ```

!!! warning "Python version"
    Python 3.10 or later is necessary. If your version is older, install
    a new version before you continue.

## Step 2: Make a virtual environment

A virtual environment prevents conflicts with other Python packages.
Use one.

=== "macOS & Linux"

    ```bash
    # Make the virtual environment
    python3 -m venv lithic

    # Activate the virtual environment
    source lithic/bin/activate
    ```

=== "Windows"

    ```powershell
    # Make the virtual environment
    python -m venv lithic

    # Let PowerShell start scripts (administrator rights can be necessary)
    Set-ExecutionPolicy Unrestricted -Scope Process

    # Activate the virtual environment
    .\lithic\Scripts\activate
    ```

=== "conda"

    ```bash
    # Make and activate a conda environment
    conda create -n lithic python=3.12
    conda activate lithic
    ```

!!! tip "Active virtual environment"
    When the virtual environment is active, the command prompt starts
    with `(lithic)`.

## Step 3: Clone the repository

Clone the Lithic Editor repository from GitHub:

```bash
git clone https://github.com/JasonGellis/lithic-editor.git
cd lithic-editor
```

### Select a branch

- **Stable**: the `main` branch has the most recent release. Use it
  unless you have a reason to use the development version.
- **Development**: the `develop` branch has the changes for the next
  release. It can be less stable.

```bash
# The stable release
git checkout main

# The development version
git checkout develop
```

## Step 4: Install Lithic Editor

Install Lithic Editor and its dependencies:

```bash
pip install .
```

This command:

- Installs the Lithic Editor package and the neural upscaling models
- Installs the dependencies, including `opencv-contrib-python`
- Makes the `lithic-editor` command available

To change the code, install in editable mode with the development
tools:

```bash
pip install -e ".[dev]"
```

To build the documentation on your computer, add the documentation
tools:

```bash
pip install -e ".[dev,docs]"
```

!!! tip "Installation without a clone"
    To install without a clone of the repository:
    ```bash
    pip install git+https://github.com/JasonGellis/lithic-editor.git
    ```

## Step 5: Make sure that the installation is correct

Show the version and the commands:

```bash
lithic-editor --version
lithic-editor --help
```

Start the graphical interface:

```bash
lithic-editor gui
```

A window with the title **Lithic Editor and Annotator** opens. If the
window opens, the installation is correct.

!!! note "macOS"
    The first start of the GUI can open a Security & Privacy dialog.
    Give the permission.

### Optional: process an example image

The repository has example drawings in `example_images/`. Process one
from the command line:

```bash
lithic-editor process example_images/369.png --output results --auto-upscale --debug
```

The result is in `results/369_cleaned.png`. The debug images show each
processing step. If there are no errors, the full pipeline operates.

### Full help

```bash
# All commands
lithic-editor --help

# All options of the process command
lithic-editor process --help

# The full help text
lithic-editor help
```

## Update Lithic Editor

To update to the latest version:

```bash
# Go to the Lithic Editor directory
cd lithic-editor

# Get the latest changes
git pull

# Install again
pip install . --upgrade
```

## Build the documentation on your computer

To read the documentation without an internet connection:

```bash
# Serve the documentation at http://127.0.0.1:8000
lithic-editor docs --offline
```

To edit and build the documentation, install the documentation tools
and start the MkDocs server:

```bash
pip install -e ".[docs]"
mkdocs serve
```

!!! tip "Online documentation"
    `lithic-editor docs` opens the online documentation in your browser.

## Installation problems

### Python version

If you get a Python version error:

```bash
# Show your Python version
python --version

# If necessary, install Python 3.10 or later with your package manager
# macOS (with Homebrew)
brew install python@3.12

# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3.12 python3.12-venv

# Windows: get Python from python.org
```

### Command not found

If `lithic-editor` is not found, the virtual environment is not active,
or the installation went into another environment:

```bash
# Activate the virtual environment, then install again
pip install .

# Or start Lithic Editor without the command
python -m lithic_editor gui
```

### Windows PowerShell execution policy

If you get an execution policy error on Windows:

```powershell
# Start PowerShell as Administrator
Set-ExecutionPolicy RemoteSigned

# Or for the current session only
Set-ExecutionPolicy Unrestricted -Scope Process
```

### Missing dependencies

If you get a missing dependency error:

```bash
# Update pip first
pip install --upgrade pip

# Then install again with full output
pip install . -v
```

### OpenCV

The neural upscaling models need the `contrib` build of OpenCV. If
OpenCV does not install, or upscaling does not operate:

```bash
# Remove all OpenCV packages
pip uninstall opencv-python opencv-python-headless opencv-contrib-python

# Install the contrib build, then Lithic Editor
pip install "opencv-contrib-python>=4.8"
pip install .
```

### The GUI does not start on Linux

Install the Qt system libraries:

```bash
# Ubuntu/Debian
sudo apt-get install libxcb-xinerama0 libxcb-cursor0 libgl1 libegl1

# Fedora
sudo dnf install python3-qt5
```

### A DLL error on Windows

Install the [Visual C++ redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)
from Microsoft. Then start Lithic Editor again.

## Remove Lithic Editor

```bash
# Remove the Lithic Editor package
pip uninstall lithic-editor

# Stop and remove the virtual environment
deactivate
rm -rf lithic/  # On Windows: rmdir /s lithic
```

## Next steps

Lithic Editor is installed. Now:

1. [Prepare your images](../user-guide/images.md)
2. [Process your first drawing](../user-guide/processing.md)
3. [Add arrows and save the result](../user-guide/arrows.md)

If you have an installation problem that is not in this guide, see the
[troubleshooting guide](../user-guide/troubleshooting.md) or [open an
issue on GitHub](https://github.com/JasonGellis/lithic-editor/issues).
