# Lithic Editor and Annotator

[![Tests](https://github.com/JasonGellis/lithic-editor/workflows/Tests/badge.svg)](https://github.com/JasonGellis/lithic-editor/actions)
[![Documentation](https://img.shields.io/badge/docs-online-blue)](https://jasongellis.github.io/lithic-editor/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)

A specialized image processing tool for archaeological lithic drawings that automatically removes ripple lines while preserving structural elements, and provides annotation capabilities for technical analysis.

📚 **[Full Documentation](https://jasongellis.github.io/lithic-editor/)** | 🐛 **[Report Issues](https://github.com/JasonGellis/lithic-editor/issues)** | 💬 **[Discussions](https://github.com/JasonGellis/lithic-editor/discussions)**

## Table of Contents

- [Overview](#overview)
  - [Features](#features)
    - [Image Processing](#image-processing)
    - [Annotation Tools](#annotation-tools)
    - [Technical Features](#technical-features)
  - [Requirements](#requirements)
  - [Installation](#installation)
    - [Option 1: Install as Package (Recommended)](#option-1-install-as-package-recommended)
    - [Option 2: Install from Git](#option-2-install-from-git)
  - [Usage Guide](#usage-guide)
    - [Processing Lithic Drawings](#processing-lithic-drawings)
    - [Annotating with Arrows](#annotating-with-arrows)
    - [Saving Results](#saving-results)
  - [Architecture](#architecture)
  - [License](#license)
  - [Acknowledgements](#acknowledgements)

## Overview

Lithic Editor and Annotator is a purpose-built application for archaeological lithic analysis. It addresses two common challenges in lithic illustration processing:

1. **Ripple Line Removal**: Automatically distinguishes and removes hatching/ripple lines from structural elements in technical lithic drawings using advanced image processing algorithms.

2. **Technical Annotation**: Provides intuitive tools for adding directional arrows to indicate striking direction for scar flakes.

Developed specifically for archaeological research, the application maintains the scientific integrity of drawings while enhancing their clarity and analytical value.

## Features

### Image Processing

- **Neural Network Upscaling**: ESPCN and FSRCNN models automatically enhance low-DPI images to 300 DPI ([Learn more](https://learnopencv.com/super-resolution-in-opencv/#sec3))
- **Intelligent Ripple Removal**: Uses graph-based analysis to identify and remove hatching lines while preserving structural features
- **Processing Visualization**: View step-by-step processing stages to understand how the algorithm works
- **Manual Editing**: Tools for touching up images before processing

### Annotation Tools

- **Directional Arrows**: Add and orient arrows to indicate force direction and flake scars
- **Arrow Customization**: Resize, rotate, and change color of arrows for precise annotation
- **Intuitive Controls**:
  - Drag to move arrows
  - Shift+drag to rotate arrows
  - Alt/Option+drag to resize arrows
- **Cross-platform**: Keyboard shortcuts adapted for both Windows/Linux and Mac

### Technical Features

- **DPI Preservation**: Maintains original image resolution and DPI information throughout processing
- **Multiple Output Formats**: Save in PNG, JPEG, or TIFF formats with preserved metadata
- **Publication-Ready Output**: Options for controlling output resolution for various uses

## Requirements

- Python 3.7+
- OpenCV with contrib modules (opencv-contrib-python) for neural network upscaling

All other dependencies are automatically installed when you install the package.

## Installation

### Option 1: Install as Package (Recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/lithic-editor.git
   cd lithic-editor
   ```

2. Install the package:
   ```bash
   pip install -e .
   ```

3. Launch the application:
   ```bash
   lithic-editor --gui
   ```

## Documentation

After installation, you can access comprehensive documentation in several ways:

### 📚 View Documentation Online
The full documentation is always available at: **https://jasongellis.github.io/lithic-editor/**

### 💻 View Documentation Locally
```bash
# Open documentation in your browser
lithic-editor docs

# Or serve documentation locally (requires mkdocs)
lithic-editor docs --offline
```

### 📖 Quick Help
```bash
lithic-editor --help     # Show all commands and options
lithic-editor help       # Show detailed help information  
lithic-editor help api   # Show API usage examples
```

The documentation includes:
- **Installation Guide** - Detailed setup instructions
- **Quick Start Tutorial** - Get processing your first image in minutes
- **User Guide** - Complete feature documentation with examples
- **Developer Guide** - Contributing, testing, and extending the application
- **API Reference** - Python API documentation for programmatic usage

## For Developers

To contribute to the project or run tests:

```bash
# Clone and install in development mode with test dependencies
git clone https://github.com/JasonGellis/lithic-editor.git
cd lithic-editor
pip install -e ".[test]"

# Run the test suite
pytest

# Run tests with coverage
pytest --cov=lithic_editor --cov-report=html

# View coverage report
open htmlcov/index.html
```

### Option 2: Install from Git

Install directly from GitHub repository:
```bash
pip install git+https://github.com/yourusername/lithic-editor.git
```

## Usage Guide

### Processing Lithic Drawings

1. **Load Image**: Click "Load Image" to open a lithic drawing file (PNG, JPEG, TIFF, BMP supported)
2. **Edit Input (Optional)**: Use the brush tools to clean up the input image if needed
3. **Process Image**: Click "Process Image" to automatically remove ripple lines
4. **View Results**: Examine the resulting image and the processing steps in the debug panel

### Annotating with Arrows

1. **Add Arrow**: Click "Add Arrow" to place an arrow in the processed image
2. **Position Arrow**: Drag the arrow to position it over a flake scar or other feature
3. **Orient Arrow**: Hold Shift and drag to rotate the arrow to indicate direction
4. **Resize Arrow**: Hold Alt (Windows/Linux) or Option (Mac) and drag to resize the arrow
5. **Change Color**: Click "Arrow Color" to select a different arrow color

### Saving Results

1. **Save Result**: Click "Save Result" to save the annotated image
2. **Format Options**: Choose from PNG, JPEG, or TIFF formats
3. **DPI Control**: The application preserves original DPI information, or you can specify DPI settings for images without metadata

## Architecture

The application consists of several key components:

- **GUI Module** (`lithic_editor.gui`): Main application window and workflow control
- **Processing Module** (`lithic_editor.processing`): Core image processing algorithms
- **Annotations Module** (`lithic_editor.annotations`): Arrow drawing and manipulation system
- **CLI Interface** (`lithic_editor.cli`): Command-line interface and help system

The ripple removal algorithm uses a multi-step approach:
1. Skeletonization of the input image
2. Graph-based analysis to identify line segments
3. Classification of segments as structural or ripple lines
4. Selective removal of ripple lines while preserving structural elements
5. Quality enhancement to produce clean, publication-ready output

## License

[MIT License](LICENSE)

## Acknowledgements

- Special thanks to The British Academemy for funding
