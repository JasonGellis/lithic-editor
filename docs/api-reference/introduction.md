# API Reference

The Lithic Editor provides multiple interfaces for integrating with your workflow:

## Available APIs

### ![](../assets/images/api.svg){: style="width:24px; height:24px; vertical-align:text-bottom; margin-right:8px; filter:brightness(0)"} [Python API](../python-api/)
Programmatic access to all processing functions for integration into your Python scripts and applications.

```python
from lithic_editor.processing import process_lithic_drawing
result = process_lithic_drawing("artifact.png")
```

### ![](../assets/images/computer.svg){: style="width:24px; height:24px; vertical-align:text-bottom; margin-right:8px; filter:brightness(0)"} [Command Line Interface](../cli-reference/)  
Complete CLI for processing images, batch operations, and automation workflows.

```bash
lithic-editor process drawing.png --output results/
```

## Quick Examples

### Basic Processing
```python
from lithic_editor.processing import process_lithic_drawing

# Simple processing
result = process_lithic_drawing("lithic_drawing.png")

# With options
result = process_lithic_drawing(
    image_path="drawing.png",
    output_folder="output",
    save_debug=True,
    output_dpi=300
)
```

### Batch Processing
```python
from pathlib import Path
from lithic_editor.processing import process_lithic_drawing

# Process all images in a directory
for image_file in Path("drawings").glob("*.png"):
    result = process_lithic_drawing(str(image_file))
    print(f"Processed: {image_file.name}")
```

### Integration Example
```python
import numpy as np
from PIL import Image
from lithic_editor.processing import process_lithic_drawing

def process_with_preprocessing(image_path):
    """Custom preprocessing before lithic processing."""
    # Load and preprocess
    img = Image.open(image_path)
    img = img.convert('L')  # Convert to grayscale
    
    # Save preprocessed image
    temp_path = "temp_preprocessed.png"
    img.save(temp_path)
    
    # Process with lithic editor
    result = process_lithic_drawing(temp_path)
    return result
```

## Return Values

The `process_lithic_drawing` function returns a dictionary containing:

```python
{
    'success': bool,           # Processing status
    'output_path': str,        # Path to processed image
    'debug_folder': str,       # Path to debug images (if save_debug=True)
    'processing_time': float, # Time taken in seconds
    'image_info': {
        'width': int,
        'height': int,
        'dpi': tuple,          # (x_dpi, y_dpi)
        'format': str          # Image format
    }
}
```

## Error Handling

```python
from lithic_editor.processing import process_lithic_drawing

try:
    result = process_lithic_drawing("drawing.png")
    if result['success']:
        print(f"Processed successfully: {result['output_path']}")
    else:
        print("Processing failed")
except FileNotFoundError:
    print("Image file not found")
except Exception as e:
    print(f"Error: {e}")
```

## Next Steps

- Explore the [Python API](../python-api/) for detailed function documentation
- Review the [CLI Reference](../cli-reference/) for command-line usage
- See [Processing Guide](../processing-images/) for workflow ideas