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
- Resolution of at least 300 DPI
- Clean, continuous lines
- Minimal noise or artifacts

## Processing Algorithm

### How It Works

The ripple removal algorithm follows these steps:

1. **Image Preprocessing**
   - Convert to grayscale if needed
   - Apply threshold to create binary image
   - Remove small noise artifacts

2. **Skeletonization**
   - Reduce lines to single-pixel width
   - Preserve connectivity and topology
   - Create network representation

3. **Line Analysis**
   - Detect individual line segments
   - Calculate orientation and length
   - Build connectivity graph

4. **Ripple Detection**
   - Identify parallel line patterns
   - Analyze spacing consistency
   - Classify as ripple or structural

5. **Selective Removal**
   - Remove identified ripple lines
   - Preserve structural boundaries
   - Maintain artifact integrity

## Using the GUI

### Step-by-Step Processing

1. **Load Your Image**
   ```
   Click "Load Image" → Select file → Open
   ```

2. **Review Input**
   - Check image quality in Input panel
   - Verify correct orientation
   - Note any problem areas

3. **Process Image**
   ```
   Click "Process Image" → Wait for completion
   ```

4. **Review Results**
   - Compare before/after
   - Check debug steps if enabled
   - Verify structural preservation

### Processing Options

#### Debug Mode
Enable to save intermediate processing steps:
- Checkbox: "Save debug images"
- Creates `image_debug/` folder
- Saves all algorithm stages

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
# Custom parameters
lithic-editor process image.png \
    --output results/ \
    --debug \
    --threshold 127 \
    --min-line-length 10
```

## Python API

### Basic Processing

```python
from lithic_editor.processing import process_lithic_drawing_improved

# Process with default settings
result = process_lithic_drawing_improved("drawing.png")

# Process with custom output
result = process_lithic_drawing_improved(
    image_path="drawing.png",
    output_folder="results/",
    save_debug=True
)
```

### Advanced Usage

```python
import numpy as np
from lithic_editor.processing import process_lithic_drawing_improved

# Process numpy array
image_array = np.array(...)  # Your image data
result = process_lithic_drawing_improved(
    image_path=image_array,
    dpi_info=(300, 300),
    save_debug=False
)

# Access processing stages
stages = result.get('debug_stages', [])
for stage_name, stage_image in stages:
    print(f"Stage: {stage_name}")
    # Process stage_image
```

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
from lithic_editor.processing import process_lithic_drawing_improved

def process_file(filename):
    return process_lithic_drawing_improved(filename)

with Pool(processes=4) as pool:
    results = pool.map(process_file, file_list)
```

## Next Steps

- Learn about [Adding Annotations](annotations.md)
- Explore [Saving Options](saving.md)
- Read [API Reference](../developer/api.md)