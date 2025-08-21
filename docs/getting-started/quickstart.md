# Quick Start Guide

This guide will help you process your first lithic drawing in under 5 minutes!

## Launch the Application

After installation, launch Lithic Editor:

```bash
lithic-editor --gui
```

The main window will open with four panels:

1. **Input Image** - Original drawing
2. **Processed Image** - Cleaned result
3. **Arrow Annotations** - Annotation canvas
4. **Processing Steps** - Debug visualization

## Step 1: Load Your Image

Click the **"Load Image"** button and select a lithic drawing file.

!!! tip "Supported Formats"
    - PNG (recommended)
    - JPEG/JPG
    - TIFF/TIF
    - BMP

The image will appear in the Input Image panel.

## Step 2: Process the Image

Click the **"Process Image"** button to remove ripple lines.

The algorithm will:
1. Analyze the drawing structure
2. Identify ripple patterns
3. Remove ripple lines
4. Preserve structural elements

!!! info "Processing Time"
    Processing typically takes 5-30 seconds depending on image size and complexity.

## Step 3: Add Annotations (Optional)

After processing, you can add directional arrows:

### Adding Arrows

1. Click **"Add Arrow"** - An arrow appears in the center
2. **Drag** to position it over a feature
3. **Shift+drag** to rotate to the desired angle
4. **Alt/Option+drag** to resize

### Customizing Arrows

- Click **"Arrow Color"** to change color
- Click **"Delete Arrow"** to remove selected arrow
- Click **"Clear Arrows"** to remove all arrows

## Step 4: Save Your Result

Click **"Save Result"** to export your processed image.

### Save Options

- **Format**: Choose PNG, JPEG, or TIFF
- **Location**: Select output folder
- **Filename**: Choose descriptive name

## Complete Example

Here's a complete workflow:

```python
# Using the Python API instead of GUI
from lithic_editor.processing import process_lithic_drawing

# Process the image
result = process_lithic_drawing(
    "lithic_artifact.png",
    output_folder="results/",
    save_debug=False  # Set True to save intermediate processing steps
)

print("Processing complete!")
```

## Tips for Best Results

### Image Quality
- Use high-resolution scans (300+ DPI)
- Ensure good contrast between lines and background
- Clean, black lines on white background work best

### Pre-Processing
If your image has issues:
1. Use the **brush tool** to clean up artifacts
2. Remove any text or scale bars before processing
3. Ensure the drawing is properly oriented

### Arrow Annotations
- Use consistent arrow sizes for professional appearance
- Match arrow direction to striking patterns
- Consider using different colors for different features

## Keyboard Shortcuts

| Action | Windows/Linux | macOS |
|--------|--------------|-------|
| Add Arrow | Ctrl+A | Cmd+A |
| Delete Arrow | Delete | Delete |
| Save Result | Ctrl+S | Cmd+S |
| Load Image | Ctrl+O | Cmd+O |
| Quit | Ctrl+Q | Cmd+Q |

## Command Line Quick Start

For batch processing or automation:

```bash
# Process a single image
lithic-editor process drawing.png

# Process with debug output
lithic-editor process drawing.png --debug

# Specify output directory
lithic-editor process drawing.png --output results/

# Process quietly (no console output)
lithic-editor process drawing.png --quiet
```

## Troubleshooting

??? question "Image doesn't load"
    - Check file format is supported
    - Ensure file isn't corrupted
    - Try converting to PNG format

??? question "Processing takes too long"
    - Reduce image size (resize to max 4000px)
    - Close other applications
    - Check available RAM

??? question "Arrows don't appear"
    - Ensure image is processed first
    - Click "Add Arrow" button
    - Check arrow color isn't same as background

## What's Next?

- Learn about [advanced processing options](../user-guide/processing.md)
- Explore [annotation techniques](../user-guide/annotations.md)
- Read the [full user guide](../user-guide/overview.md)

## Getting Help

If you encounter issues:

1. Check the [User Guide](../user-guide/overview.md)
2. Search [GitHub Issues](https://github.com/JasonGellis/lithic-editor/issues)
3. Ask in [Discussions](https://github.com/JasonGellis/lithic-editor/discussions)
4. Contact: jg760@cam.ac.uk