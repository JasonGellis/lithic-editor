# Python API

Complete reference for the Lithic Editor Python API.

## Core Functions

### process_lithic_drawing

The main processing function for removing ripple lines from lithic drawings.

```python
process_lithic_drawing(
    image_path: str,
    output_folder: str = "image_debug",
    dpi_info: Optional[Tuple[int, int]] = None,
    format_info: Optional[str] = None,
    output_dpi: Optional[int] = None,
    save_debug: bool = False,
    upscale_low_dpi: bool = False,
    default_dpi: Optional[int] = None,
    upscale_model: str = 'espcn',
    target_dpi: int = 300,
    scale_image_path: Optional[str] = None,
    return_scale_factor: bool = False,
    debug_filename: Optional[str] = None,
    preserve_cortex: bool = True
) -> np.ndarray
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image_path` | `str` | *required* | Path to the input image file |
| `output_folder` | `str` | `"image_debug"` | Directory for output files |
| `dpi_info` | `tuple` | `None` | Override DPI as (x_dpi, y_dpi) |
| `format_info` | `str` | `None` | Override output format (png, jpg, tiff) |
| `output_dpi` | `int` | `None` | Set specific output DPI |
| `save_debug` | `bool` | `False` | Save intermediate processing steps to disk |
| `upscale_low_dpi` | `bool` | `False` | Enable neural network upscaling for low-DPI images |
| `default_dpi` | `int` | `None` | DPI to assume if metadata missing |
| `upscale_model` | `str` | `'espcn'` | Model to use: 'espcn' or 'fsrcnn' |
| `target_dpi` | `int` | `300` | Target DPI for upscaling |
| `scale_image_path` | `str` | `None` | Scale image to process with same factor |
| `return_scale_factor` | `bool` | `False` | Return upscaling details in result |
| `debug_filename` | `str` | `None` | Custom filename for debug images |
| `preserve_cortex` | `bool` | `True` | Preserve cortex stippling (default: enabled) |

#### Returns

**Basic mode:** Returns `numpy.ndarray` containing the processed image.

**Extended mode** (when `return_scale_factor=True` or `scale_image_path` provided): Returns dictionary containing:
- `processed_image` (np.ndarray): The processed image array
- `scale_factor` (float): Upscaling factor applied (1.0 if no scaling)
- `original_dpi` (int): Original image DPI
- `final_dpi` (int): Final image DPI after processing
- `processed_scale` (np.ndarray): Processed scale image (if provided)

#### Algorithm Features

The function automatically adapts processing parameters based on image DPI:

**DPI-Aware Y-tip Removal:**
- **600+ DPI:** 8-pixel threshold for Y-tip junction detection
- **300-599 DPI:** 5-pixel threshold
- **150-299 DPI:** 3-pixel threshold
- **<150 DPI:** 2-pixel threshold (conservative to preserve structural details)

**DPI-Aware Thickness Reconstruction:**
- **300+ DPI:** 4-6 pixel thickness preservation
- **150-299 DPI:** 3-4 pixel thickness preservation
- **<150 DPI:** 1-2 pixel thickness preservation

**DPI-Adaptive Cortex Filtering:**
- Minimum/maximum thresholds scale quadratically with DPI
- Filters noise while preserving legitimate cortex stippling

#### Example Usage

```python
from lithic_editor.processing import process_lithic_drawing

# Basic usage
result = process_lithic_drawing("artifact.png")

# With all options including upscaling
result = process_lithic_drawing(
    image_path="drawing.png",
    output_folder="results",
    dpi_info=(300, 300),
    format_info="png",
    output_dpi=300,
    save_debug=True,
    upscale_low_dpi=True,
    default_dpi=150,
    upscale_model='fsrcnn',
    target_dpi=300,
    preserve_cortex=True
)

# Check results
if result['success']:
    print(f"Output: {result['output_path']}")
    print(f"Processing time: {result['processing_time']:.2f}s")
```

## Annotation Classes

### Arrow

Class for managing directional arrows in lithic drawings.

```python
from lithic_editor.annotations import Arrow

# Create an arrow
arrow = Arrow(
    start_point=(100, 100),
    end_point=(200, 150),
    color='black',
    width=2
)

# Add to image
arrow.draw(image)
```

#### Methods

##### `__init__(start_point, end_point, color='black', width=2)`
Initialize an arrow annotation.

##### `draw(image) -> np.ndarray`
Draw the arrow on an image.

##### `rotate(angle: float, center: tuple) -> None`
Rotate the arrow around a center point.

##### `scale(factor: float) -> None`
Scale the arrow size.

## Image Processing Utilities

### load_image

Load and validate an image file.

```python
from lithic_editor.utils import load_image

image, metadata = load_image("drawing.png")
print(f"Image shape: {image.shape}")
print(f"DPI: {metadata.get('dpi', 'Not set')}")
```

### save_with_metadata

Save an image preserving metadata.

```python
from lithic_editor.utils import save_with_metadata

save_with_metadata(
    image_array,
    output_path="result.png",
    dpi=(300, 300),
    format="PNG"
)
```

## Advanced Usage

### Custom Processing Pipeline

```python
from lithic_editor.processing import (
    process_lithic_drawing,
    apply_skeleton,
    detect_ripples
)
from lithic_editor.utils import load_image, save_with_metadata

def custom_pipeline(image_path):
    """Custom processing with intermediate steps."""
    
    # Load image
    image, metadata = load_image(image_path)
    
    # Step 1: Create skeleton
    skeleton = apply_skeleton(image)
    
    # Step 2: Detect ripple patterns
    ripples = detect_ripples(skeleton)
    
    # Step 3: Process with custom parameters
    result = process_lithic_drawing(
        image_path,
        save_debug=True,
        output_dpi=metadata.get('dpi', [300, 300])[0]
    )
    
    return result
```

### Batch Processing with Progress

```python
from pathlib import Path
from lithic_editor.processing import process_lithic_drawing
from tqdm import tqdm  # Optional: for progress bar

def batch_process(input_dir, output_dir):
    """Process all images in a directory."""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Get all image files
    image_files = list(input_path.glob("*.png"))
    image_files.extend(input_path.glob("*.jpg"))
    
    results = []
    for image_file in tqdm(image_files, desc="Processing"):
        try:
            result = process_lithic_drawing(
                str(image_file),
                output_folder=str(output_path / image_file.stem)
            )
            results.append({
                'file': image_file.name,
                'success': result['success'],
                'time': result['processing_time']
            })
        except Exception as e:
            results.append({
                'file': image_file.name,
                'success': False,
                'error': str(e)
            })
    
    return results

# Usage
results = batch_process("drawings/", "processed/")
print(f"Processed {sum(r['success'] for r in results)}/{len(results)} images")
```

### Integration with NumPy/PIL

```python
import numpy as np
from PIL import Image
from lithic_editor.processing import process_lithic_drawing

def process_from_array(image_array: np.ndarray) -> np.ndarray:
    """Process a NumPy array."""
    
    # Save array as temporary image
    temp_path = "temp_image.png"
    Image.fromarray(image_array).save(temp_path)
    
    # Process
    result = process_lithic_drawing(temp_path)
    
    # Load result as array
    if result['success']:
        processed = np.array(Image.open(result['output_path']))
        return processed
    return image_array

# Example with PIL
pil_image = Image.open("drawing.png")
array = np.array(pil_image)
processed_array = process_from_array(array)
result_image = Image.fromarray(processed_array)
```

## Error Handling

### Common Exceptions

```python
from lithic_editor.processing import process_lithic_drawing

try:
    result = process_lithic_drawing("image.png")
except FileNotFoundError:
    print("Image file not found")
except ValueError as e:
    print(f"Invalid parameter: {e}")
except PermissionError:
    print("Cannot write to output directory")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Validation

```python
from pathlib import Path

def validate_and_process(image_path):
    """Validate before processing."""
    
    path = Path(image_path)
    
    # Check file exists
    if not path.exists():
        raise FileNotFoundError(f"File not found: {image_path}")
    
    # Check file type
    valid_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.tif'}
    if path.suffix.lower() not in valid_extensions:
        raise ValueError(f"Unsupported format: {path.suffix}")
    
    # Check file size
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > 100:
        raise ValueError(f"File too large: {size_mb:.1f}MB")
    
    # Process
    return process_lithic_drawing(str(path))
```

## Performance Tips

1. **Memory Management**: For large images, process in batches
2. **Parallel Processing**: Use multiprocessing for batch operations
3. **Cache Results**: Store processed images to avoid reprocessing
4. **Optimize Input**: Resize very large images before processing

```python
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

def parallel_batch_process(image_files, max_workers=4):
    """Process multiple images in parallel."""
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for image_file in image_files:
            future = executor.submit(process_lithic_drawing, str(image_file))
            futures.append((image_file, future))
        
        results = []
        for image_file, future in futures:
            try:
                result = future.result(timeout=60)
                results.append((image_file, result))
            except Exception as e:
                print(f"Failed to process {image_file}: {e}")
                
    return results
```