# Processing Images

## Overview

The image processing engine in Lithic Editor uses advanced algorithms to automatically identify and remove ripple lines while preserving the structural elements of lithic drawings.

## Loading Images

### Supported Formats
- **PNG** (recommended): Lossless compression, transparency support
- **JPEG/JPG**: Widely compatible, smaller file sizes
- **TIFF/TIF**: Professional quality, uncompressed
- **BMP**: Simple format, uncompressed

### Image Requirements
For best results, your images should have:
- High contrast (black lines on white background)
- Resolution of at least 300 DPI (automatic upscaling available for lower DPI)
- Clean, continuous lines
- Minimal noise or artifacts

### Neural Network Upscaling
Lithic Editor includes ESPCN and FSRCNN neural network models to automatically upscale low-DPI images:
- **ESPCN**: Efficient Sub-Pixel CNN, faster processing
- **FSRCNN**: Fast Super-Resolution CNN, higher quality
- **Automatic DPI detection** from image metadata
- **User dialogs** for DPI selection when metadata is missing
- **300 DPI target** for optimal processing results

For technical details about these super-resolution models, see [OpenCV Super Resolution Tutorial](https://learnopencv.com/super-resolution-in-opencv/#sec3).

## Processing Algorithm

### How It Works

The ripple removal algorithm follows these steps:

1. **Neural Network Upscaling** (if needed)
   - Detect DPI from image metadata or prompt user
   - Upscale low-DPI images using ESPCN/FSRCNN models
   - Maintain aspect ratio and line quality

2. **Image Preprocessing**
   - Convert to grayscale if needed
   - Apply threshold to create binary image
   - Remove small noise artifacts

3. **Skeletonization**
   - Reduce lines to single-pixel width
   - Preserve connectivity and topology
   - Create network representation

4. **Line Analysis**
   - Detect individual line segments
   - Calculate orientation and length
   - Build connectivity graph

5. **Ripple Detection**
   - Identify parallel line patterns
   - Analyze spacing consistency
   - Classify as ripple or structural

6. **Selective Removal**
   - Remove identified ripple lines
   - Preserve structural boundaries
   - Maintain artifact integrity

7. **Endpoint Filtering**
   - Refine endpoint decisions after cleaning
   - Remove artifacts created by ripple removal
   - Preserve important structural endpoints

## Using the GUI

### Step-by-Step Processing

1. **Load Your Image**
   ```
   Click "Load Image" → Select file → Open
   ```

2. **DPI Detection and Upscaling**
   - System automatically detects DPI from image metadata
   - If missing, dialog prompts for DPI selection (72, 96, 150, 200 or custom)
   - If below 300 DPI, upscaling dialog offers ESPCN/FSRCNN options

3. **Review Input**
   - Check image quality in Input panel
   - Verify correct orientation
   - Note any problem areas

4. **Process Image**
   ```
   Click "Process Image" → Wait for completion
   ```

5. **Review Results**
   - Compare before/after
   - Check debug steps if enabled
   - Verify structural preservation

### Processing Options

#### Debug Mode
Enable to view and save intermediate processing steps:
- Checkbox: "View and Save Debug Images"
- Shows processing steps in the Processing Steps panel
- Creates `image_debug/` folder with all algorithm stages

#### Quality Settings
Adjust processing parameters:
- Line thickness tolerance
- Ripple pattern sensitivity
- Structural preservation level

## Command Line Processing

### Basic Usage

```bash
# Process single image
lithic-editor process input.png

# Specify output directory
lithic-editor process input.png --output results/

# Enable debug mode
lithic-editor process input.png --debug
```

### Batch Processing

```bash
# Process all PNG files
for file in *.png; do
    lithic-editor process "$file" --output processed/
done

# Process with parallel jobs
find . -name "*.png" | parallel lithic-editor process {} --quiet
```

### Advanced Options

```bash
# Neural network upscaling with debug output
lithic-editor process low_res.png \
    --auto-upscale \
    --default-dpi 150 \
    --upscale-model fsrcnn \
    --upscale-threshold 300 \
    --debug

# Batch processing with upscaling
for file in *.png; do
    lithic-editor process "$file" \
        --auto-upscale \
        --default-dpi 200 \
        --output "processed/${file%.png}/"
done
```

## Python API

### Using the Python API

For detailed API documentation and examples, see the [API Reference](../api-reference/overview.md).

```python
from lithic_editor.processing import process_lithic_drawing

# Basic example
result = process_lithic_drawing("drawing.png", save_debug=True)

# With neural network upscaling
result = process_lithic_drawing(
    "low_dpi.png",
    save_debug=True,
    upscale_low_dpi=True,
    default_dpi=150,
    upscale_model='espcn',
    target_dpi=300
)
```

For advanced usage, batch processing, and integration examples, refer to:
- [Python API Documentation](../api-reference/python-api.md)
- [CLI Reference](../api-reference/cli-reference.md)

## Troubleshooting

### Common Issues

??? problem "Structural lines are removed"
    **Causes:**
    - Lines too similar to ripple pattern
    - Incorrect threshold settings
    
    **Solutions:**
    - Increase structural preservation setting
    - Manually edit before processing
    - Use debug mode to identify issue

??? problem "Ripples not fully removed"
    **Causes:**
    - Inconsistent ripple pattern
    - Poor image quality
    - Ripples too thick
    
    **Solutions:**
    - Improve scan quality
    - Pre-process to enhance contrast
    - Adjust sensitivity settings

??? problem "Processing takes too long"
    **Causes:**
    - Image too large
    - Insufficient memory
    - Complex line patterns
    
    **Solutions:**
    - Resize image to 3000px max
    - Close other applications
    - Use batch processing overnight

### Image Preparation Tips

1. **Scanning**
   - Use 300+ DPI
   - Black and white mode
   - Clean scanner glass

2. **Editing**
   - Remove text and scales
   - Fill gaps in lines
   - Increase contrast

3. **Format**
   - Save as PNG
   - Use lossless compression
   - Preserve metadata

## Best Practices

### Quality Control

!!! tip "Always Review Debug Images"
    Enable debug mode for important images to verify the algorithm isn't removing structural elements.

### Workflow Optimization

1. **Test on Sample**
   - Process small section first
   - Adjust settings as needed
   - Apply to full image

2. **Batch Similar Images**
   - Group by drawing style
   - Use consistent settings
   - Review results together

3. **Archive Originals**
   - Keep unprocessed versions
   - Document processing parameters
   - Note any manual edits

## Performance Optimization

### Memory Management
- Process images under 4000×4000 pixels
- Close unnecessary applications
- Use 64-bit Python

### Speed Improvements
- Resize large images first
- Process in batches overnight
- Use SSD for temp files

### Parallel Processing
```python
from multiprocessing import Pool
from lithic_editor.processing import process_lithic_drawing

def process_file(filename):
    return process_lithic_drawing(filename)

with Pool(processes=4) as pool:
    results = pool.map(process_file, file_list)
```

## Next Steps

- Learn about [Adding Annotations](annotations.md)
- Explore [Saving Options](saving.md)
- Read [API Reference](../developer/api.md)