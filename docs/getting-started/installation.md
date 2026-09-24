# Installation

Lithic Editor runs on Windows, macOS and Linux with Python 3.10 to 3.13. The steps are the same on each system. Where a command is different, select the tab for your system.

## Step 1: Install Python and Git

=== "Windows"
    1. Download Python 3.12 from [python.org](https://www.python.org/downloads/windows/) and start the installer.
    2. Set **Add python.exe to PATH** on the first page of the installer. Then click **Install Now**.
    3. Download Git from [git-scm.com](https://git-scm.com/download/win) and start the installer. Keep the default options.
    4. Open **PowerShell** from the Start menu. Use PowerShell for all the commands on this page.

    If you use `winget`, these two commands do the same:
    ```powershell
    winget install Python.Python.3.12
    winget install Git.Git
    ```
    Then close and open PowerShell again.

=== "macOS"
    1. Open **Terminal** from Applications > Utilities. Use Terminal for all the commands on this page.
    2. Install the command line tools. This installs Git:
       ```bash
       xcode-select --install
       ```
    3. Install Python 3.12. With [Homebrew](https://brew.sh):
       ```bash
       brew install python@3.12
       ```
       Or download the installer from [python.org](https://www.python.org/downloads/macos/) and start it.

=== "Linux"
    Open a terminal. Use it for all the commands on this page. Install Python, pip, venv and Git with your package manager:

    ```bash
    # Ubuntu / Debian
    sudo apt-get update
    sudo apt-get install python3 python3-pip python3-venv git

    # Fedora
    sudo dnf install python3 python3-pip git

    # Arch
    sudo pacman -S python python-pip git
    ```

    The GUI needs the Qt system libraries:
    ```bash
    # Ubuntu / Debian
    sudo apt-get install libxcb-xinerama0 libxcb-cursor0 libgl1 libegl1
    ```

Check the installation:

=== "Windows"
    ```powershell
    python --version    # Shows 3.10 or later
    git --version
    ```
    If `python` is not found, use `py -3 --version`.

=== "macOS"
    ```bash
    python3 --version   # Shows 3.10 or later
    git --version
    ```

=== "Linux"
    ```bash
    python3 --version   # Shows 3.10 or later
    git --version
    ```

## Step 2: Make an environment

An environment keeps the Lithic Editor packages separate from other Python programs. Make one and activate it. The prompt then shows the environment name.

=== "Windows"
    ```powershell
    python -m venv lithic-env
    lithic-env\Scripts\activate
    ```
    If PowerShell does not permit the activate script, run this once:
    ```powershell
    Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
    ```

=== "macOS"
    ```bash
    python3 -m venv lithic-env
    source lithic-env/bin/activate
    ```

=== "Linux"
    ```bash
    python3 -m venv lithic-env
    source lithic-env/bin/activate
    ```

=== "conda (any system)"
    ```bash
    conda create -n lithic python=3.12
    conda activate lithic
    ```

Activate the environment each time you open a new terminal before you use Lithic Editor.

## Step 3: Install Lithic Editor

With the environment active, select one method.

### Method 1: Install from GitHub

For most users.

```bash
pip install git+https://github.com/JasonGellis/lithic-editor.git
```

To install a tagged release, add `@<tag>` to the end of the URL.

### Method 2: Development installation

For users who change the code.

```bash
git clone https://github.com/JasonGellis/lithic-editor.git
cd lithic-editor

pip install -e .              # the package
pip install -e ".[dev]"       # the package, the tests and ruff
pip install -e ".[docs]"      # the package and the documentation tools
pip install -e ".[dev,docs]"  # all tools
```

pip installs all dependencies with the package. This includes `opencv-contrib-python`, which the neural upscaling models need.

## Step 4: Check the installation

```bash
lithic-editor --version   # Shows the version
lithic-editor --help      # Shows the commands
lithic-editor gui         # Starts the GUI
```

On macOS, the first start of the GUI can open a Security & Privacy dialog. Give the permission.

## Dependencies

### Core dependencies

| Package | Purpose |
|---------|---------|
| numpy | Array operations |
| opencv-contrib-python | Image processing and the neural upscaling models |
| Pillow | Image input and output |
| PyQt5 | GUI framework |
| scikit-image | Thresholding and skeletonization |
| networkx | Graph analysis |
| scipy | Scientific computing |
| PyYAML | The configuration file |

!!! note "opencv-contrib-python"
    The neural upscaling models need the `contrib` build of OpenCV. The plain `opencv-python` package does not include them.

### Optional dependencies

| Group | Purpose | Packages |
|-------|---------|----------|
| `test` | Run the tests | pytest, pytest-qt, pytest-cov |
| `dev` | Run the tests and the linter | `test` packages and ruff |
| `docs` | Build the documentation | mkdocs, mkdocs-material, mkdocs-material-extensions, pymdown-extensions |

You do not need the `docs` packages to read the documentation. Run `lithic-editor docs` to open the online documentation.

## Troubleshooting

??? failure "'lithic-editor' is not recognized / command not found"
    The environment is not active, or the install went into another environment.
    Activate the environment (Step 2) and run `pip install` again. Or start the program with `python -m lithic_editor`.

??? failure "ImportError: No module named 'PyQt5'"
    Install PyQt5 by hand:
    ```bash
    pip install PyQt5
    ```

??? failure "OpenCV import error, or no neural upscaling"
    Remove all OpenCV packages. Then install `opencv-contrib-python`:
    ```bash
    pip uninstall opencv-python opencv-python-headless opencv-contrib-python
    pip install opencv-contrib-python
    ```

??? failure "The GUI does not start on Linux"
    Install the Qt system libraries:
    ```bash
    # Ubuntu / Debian
    sudo apt-get install libxcb-xinerama0 libxcb-cursor0 libgl1 libegl1

    # Fedora
    sudo dnf install python3-qt5
    ```

??? failure "Permission denied"
    Install in your user directory:
    ```bash
    pip install --user git+https://github.com/JasonGellis/lithic-editor.git
    ```

??? failure "Windows: a DLL error when the GUI starts"
    Install the [Visual C++ redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) from Microsoft.

## Update

```bash
# Installed from GitHub
pip install --upgrade git+https://github.com/JasonGellis/lithic-editor.git

# Development installation
cd lithic-editor
git pull
pip install -e . --upgrade
```

## Uninstall

```bash
pip uninstall lithic-editor
```

## Next Steps

Go to the [User Guide](../user-guide/overview.md) to process your first drawing.
