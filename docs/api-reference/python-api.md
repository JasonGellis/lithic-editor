# Python API

This page describes the public Python functions of the Lithic Editor.

## process_lithic_drawing

`process_lithic_drawing` removes the ripple lines from one lithic drawing. It keeps the scar contours, the outlines and the cortex stipple.

```python
from lithic_editor.processing import process_lithic_drawing

process_lithic_drawing(
    image_path,
    output_folder="image_debug",
    dpi_info=None,
    format_info=None,
    output_dpi=None,
    save_debug=False,
    upscale_low_dpi=False,
    default_dpi=None,
    upscale_model="espcn",
    scale_image_path=None,
    return_scale_factor=False,
    debug_filename=None,
    preserve_cortex=True,
    max_upscale_factor=4,
    restore_original_size=True,
    smooth_lines=None,
    config=None,
)
```

The function does not write the result to a file. It returns the result as a numpy array. The function prints progress messages to standard output.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image_path` | `str` or `numpy.ndarray` | required | The path of the input image, or a grayscale array with two dimensions. |
| `output_folder` | `str` | `"image_debug"` | The directory for the debug images. The function makes the directory only when `save_debug` is `True`. |
| `dpi_info` | `tuple` or `int` | `None` | The DPI of the input, as `(x_dpi, y_dpi)` or one integer. If `None`, the function reads the DPI tag of the file. An array has no DPI tag. |
| `format_info` | `str` | `None` | The file format for the debug images. If `None`, the function uses the format of the input file. |
| `output_dpi` | `int` | `None` | The DPI tag written to the debug images. If `None`, the function writes the input DPI. |
| `save_debug` | `bool` | `False` | Write the debug images to `output_folder`. |
| `upscale_low_dpi` | `bool` | `False` | Permit upscaling for processing when the lines are too thin or too close. See [Upscaling](#upscaling). |
| `default_dpi` | `int` | `None` | The DPI to use when the input has no DPI tag. The pipeline uses the DPI to scale its size thresholds. |
| `upscale_model` | `str` | configuration (`"espcn"`) | The neural model for upscaling: `"espcn"` or `"fsrcnn"`. Both models come with the package. |
| `scale_image_path` | `str` | `None` | The path of the scale bar image scanned with the drawing. See [Scale image](#scale-image). |
| `return_scale_factor` | `bool` | `False` | Return a dictionary with the image and the scale data. See [Return value](#return-value). |
| `debug_filename` | `str` | `None` | The base name of the debug images. If `None`, the function uses the stem of the input file, or `image` for an array. |
| `preserve_cortex` | `bool` | `True` | Separate the cortex stipple from the lines and keep it. If `False`, the pipeline processes the stipple as lines. |
| `max_upscale_factor` | `int` | configuration (`4`) | The largest permitted upscale factor. |
| `restore_original_size` | `bool` | `True` | Return the result at the pixel size of the input. If `False`, keep the upscaled working size. |
| `smooth_lines` | `bool` | configuration (`True`) | Apply a Gaussian blur before the threshold. The sigma is a fraction of the measured line width. |
| `config` | `Config`, `str` or path | `None` | The configuration. `None` reads the file named by `LITHIC_EDITOR_CONFIG`, or the shipped file. See [Configuration](../user-guide/configuration.md). |

### Configuration

Every size threshold comes from the configuration file. The parameters `upscale_model`, `max_upscale_factor` and `smooth_lines` take their default from it. An explicit argument replaces the configuration value.

```python
from lithic_editor.config import load_config
from lithic_editor.processing import process_lithic_drawing

config = load_config("my_config.yaml")
result = process_lithic_drawing("drawing.png", config=config)
```

### DPI

The pipeline does not guess a DPI. The DPI comes from one of these sources, in this order:

1. `dpi_info`.
2. The DPI tag of the input file.
3. `default_dpi`.

If no source gives a DPI, the pipeline uses fixed size thresholds. The returned `original_dpi` and `final_dpi` are then `None`.

### Upscaling

The pipeline measures the line width and the hatch gap of the drawing in pixels. The floors for processing are a line width of 6 px and a hatch gap of 12 px.

When `upscale_low_dpi` is `True` and a measured value is below its floor, the pipeline upscales the image for processing. The factor is the smallest of 2, 3 or 4 that lifts both values above their floors. The factor is never more than `max_upscale_factor`. The pipeline never downscales.

By default the pipeline returns the result at the input pixel size and DPI. Set `restore_original_size=False` to keep the upscaled size. The DPI of the result is then the input DPI multiplied by the factor.

### Scale image

Give `scale_image_path` when a scale bar was scanned with the drawing. The function reads the scale image as grayscale and returns it in the `processed_scale` key of the result.

- With `restore_original_size=True` (the default), the scale image is not changed.
- With `restore_original_size=False`, the scale image is upscaled by the same factor as the drawing. The drawing and the scale bar keep one pixel scale.

If the scale image cannot be read, the function prints a message and the result has no `processed_scale` key.

### Return value

**Default mode.** The function returns a `numpy.ndarray` of type `uint8`. Black pixels (0) are lines. White pixels (255) are background. The array has the shape of the input when `restore_original_size` is `True`.

**Dictionary mode.** When `return_scale_factor` is `True` or `scale_image_path` is given, the function returns a dictionary:

| Key | Type | Description |
|-----|------|-------------|
| `processed_image` | `numpy.ndarray` | The cleaned drawing, black lines on white. |
| `scale_factor` | `int` | The size of the returned image relative to the input. The value is 1 when the result is at the input size. |
| `working_scale_factor` | `int` | The factor used for processing. The value is 1 when the pipeline did not upscale. |
| `original_dpi` | `int` or `None` | The DPI of the input. |
| `final_dpi` | `int` or `None` | The DPI of the returned image. The value equals `original_dpi` multiplied by `scale_factor`. |
| `processed_scale` | `numpy.ndarray` | The scale bar image. The key is present only when `scale_image_path` was given and read. |

### Debug images

When `save_debug` is `True`, the function writes one PNG file for each step to `output_folder`. `<name>` is the value of `debug_filename` or the stem of the input file.

| File | Content |
|------|---------|
| `0_<name>_input.png` | The input. Written only when the pipeline upscaled. |
| `0a_<name>_upscaled.png` | The upscaled input. Written only when the pipeline upscaled. |
| `0b_<name>_upscaled_scale.png` | The upscaled scale image. Written only with a scale image and `restore_original_size=False`. |
| `1_<name>_original_image.png` | The grayscale image at the working size. |
| `1b_<name>_smoothed.png` | The image after the Gaussian blur. |
| `1c_<name>_binary_thresholded.png` | The binary image after the Sauvola threshold. |
| `2a_<name>_structural_only.png` | The lines without the cortex stipple. |
| `2b_<name>_cortex_mask.png` | The cortex stipple. |
| `3_<name>_skeleton.png` | The skeleton of the lines. |
| `4_<name>_endpoints_junctions.png` | The endpoints and junctions of the skeleton. |
| `5_<name>_labeled_segments.png` | The skeleton segments, each with a label. |
| `6_<name>_ripple_identification.png` | Red = ripple, white = structural, green dots = junctions. |
| `7_<name>_skeleton_cleaned.png` | The skeleton without the ripple segments. |
| `7a_<name>_endpoint_filtering.png` | The skeleton after the endpoint filter. |
| `8_<name>_final_cleaned.png` | The cleaned drawing at the working size. |
| `9_<name>_restored.png` | The cleaned drawing at the input size. Written only when the pipeline upscaled and restored the size. |

### Errors

| Exception | Cause |
|-----------|-------|
| `ValueError` | The input file cannot be read. This includes a file that does not exist. |

## Examples

### Basic

```python
from PIL import Image
from lithic_editor.processing import process_lithic_drawing

cleaned = process_lithic_drawing("drawing.png")
Image.fromarray(cleaned).save("drawing_cleaned.png")
```

### With upscaling

Permit upscaling and read the scale data from the result. The result is at the input size and DPI.

```python
from PIL import Image
from lithic_editor.processing import process_lithic_drawing

result = process_lithic_drawing(
    "drawing_150dpi.png",
    upscale_low_dpi=True,
    default_dpi=150,
    return_scale_factor=True,
)

print(f"Processed at {result['working_scale_factor']}x")
print(f"Returned at {result['scale_factor']}x, {result['final_dpi']} DPI")

image = Image.fromarray(result["processed_image"])
dpi = result["final_dpi"]
if dpi:
    image.save("drawing_cleaned.png", dpi=(dpi, dpi))
else:
    image.save("drawing_cleaned.png")
```

### Keep the upscaled size with a scale image

Keep the working size and scale the scale bar by the same factor. Write the same DPI tag to both files.

```python
from PIL import Image
from lithic_editor.processing import process_lithic_drawing

result = process_lithic_drawing(
    "drawing.png",
    upscale_low_dpi=True,
    restore_original_size=False,
    scale_image_path="scale_bar.png",
)

dpi = result["final_dpi"]
tag = {"dpi": (dpi, dpi)} if dpi else {}
Image.fromarray(result["processed_image"]).save("drawing_cleaned.png", **tag)
if "processed_scale" in result:
    Image.fromarray(result["processed_scale"]).save("drawing_scale.png", **tag)
```

### Numpy array input

Give a grayscale array instead of a path. An array has no DPI tag, so give `dpi_info`.

```python
import numpy as np
from PIL import Image
from lithic_editor.processing import process_lithic_drawing

gray = np.array(Image.open("drawing.png").convert("L"))

cleaned = process_lithic_drawing(
    gray,
    dpi_info=300,
    debug_filename="drawing",
)
```

### Batch loop

Process each PNG file in a directory. Write each result to an output directory.

```python
from pathlib import Path
from PIL import Image
from lithic_editor.processing import process_lithic_drawing

input_dir = Path("drawings")
output_dir = Path("processed")
output_dir.mkdir(exist_ok=True)

for path in sorted(input_dir.glob("*.png")):
    result = process_lithic_drawing(
        str(path),
        upscale_low_dpi=True,
        return_scale_factor=True,
    )
    dpi = result["final_dpi"]
    tag = {"dpi": (dpi, dpi)} if dpi else {}
    Image.fromarray(result["processed_image"]).save(
        output_dir / f"{path.stem}_cleaned.png", **tag
    )
    print(f"{path.name}: processed at {result['working_scale_factor']}x")
```

## Resolution helpers

The module `lithic_editor.processing.resolution` has the functions that measure the drawing and choose the upscale factor.

### measure_line_geometry

```python
from lithic_editor.processing.resolution import measure_line_geometry

measure_line_geometry(gray) -> LineGeometry
```

`gray` is a grayscale numpy array. The function returns a `LineGeometry` object.

### LineGeometry

| Attribute | Type | Description |
|-----------|------|-------------|
| `line_width` | `float` | The typical line width in pixels. `NaN` when the function cannot measure it. |
| `hatch_gap` | `float` | The typical gap between lines in pixels. `NaN` when the function cannot measure it. |
| `ink_fraction` | `float` | The fraction of pixels that are ink, from 0.0 to 1.0. |

`LineGeometry.describe()` returns a text such as `line width 4.7 px, hatch gap 9.0 px`.

### choose_upscale_factor

```python
from lithic_editor.processing.resolution import choose_upscale_factor

choose_upscale_factor(geometry, max_factor=4) -> int
```

The function returns the smallest factor of 1, 2, 3 or 4 that lifts both values to their floors. The floors are a line width of 6 px and a hatch gap of 12 px. The factor is never more than `max_factor`. The function returns 1 when no upscaling is necessary or when nothing was measured.

```python
import numpy as np
from PIL import Image
from lithic_editor.processing.resolution import measure_line_geometry, choose_upscale_factor

gray = np.array(Image.open("drawing.png").convert("L"))
geometry = measure_line_geometry(gray)
factor = choose_upscale_factor(geometry)
print(f"{geometry.describe()}: factor {factor}x")
```
