# Lithic Editor and Annotator

A specialized image processing tool for archaeological lithic drawings that automatically removes ripple lines while preserving structural elements, and provides annotation capabilities for technical analysis.

## Overview

Lithic Editor and Annotator is a purpose-built application for archaeological lithic analysis. It addresses two common challenges in lithic illustration processing:

1. **Ripple Line Removal**: Automatically distinguishes and removes hatching/ripple lines from structural elements in technical lithic drawings using advanced image processing algorithms.

2. **Technical Annotation**: Provides intuitive tools for adding directional arrows to indicate striking direction for scar flakes.

Developed specifically for archaeological research, the application maintains the scientific integrity of drawings while enhancing their clarity and analytical value.

## Features

### Image Processing

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
- PyQt5
- OpenCV (cv2)
- NumPy
- Pillow (PIL)
- scikit-image
- NetworkX
- SciPy

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/lithic-editor-annotator.git
   cd lithic-editor-annotator
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   python lithic_GUI.py
   ```

## Usage Guide

### Processing Lithic Drawings

1. **Load Image**: Click "Load Image" to open a lithic drawing file (PNG, JPEG, TIFF, BMP supported)
2. **Edit Input (Optional)**: Use the brush tools to clean up the input image if needed
3. **Process Image**: Click "Process Image" to automatically remove ripple lines
4. **View Results**: Examine the resulting image and the processing steps in the debug panel

### Annotating with Arrows

1. **Add Arrow**: Click "Add Arrow" to place an arrow in the center of the processed image
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

- **Main GUI** (`lithic_GUI.py`): Main application window and workflow control
- **Ripple Removal Engine** (`ripple_remover.py`): Core image processing algorithms
- **Arrow Annotation** (`arrow_annotations.py`, `arrow_integration.py`): Arrow drawing and manipulation system

The ripple removal algorithm uses a multi-step approach:
1. Skeletonization of the input image
2. Graph-based analysis to identify line segments
3. Classification of segments as structural or ripple lines
4. Selective removal of ripple lines while preserving structural elements
5. Quality enhancement to produce clean, publication-ready output

## License

[MIT License](LICENSE)

## Acknowledgements

- Developed for archaeological research applications
- Special thanks to The British Academemy for funding
