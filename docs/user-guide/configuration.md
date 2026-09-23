# Configuration

Every size threshold of the pipeline is in one configuration file. You can change a value without a change to the code.

## The default file

The package includes the default file: `lithic_editor/config/config.yaml`. Each key has a comment that explains it. To find the file on your computer:

```python
from lithic_editor.config import default_config_path
print(default_config_path())
```

## Use your own file

1. Copy the default file.
2. Change the values you want. Remove the other keys, or keep them.
3. Give the path of your file:

=== "Command line"
    ```bash
    lithic-editor process drawing.png --config my_config.yaml
    ```

=== "Python"
    ```python
    from lithic_editor.processing import process_lithic_drawing

    result = process_lithic_drawing("drawing.png", config="my_config.yaml")
    ```

=== "Environment variable"
    ```bash
    export LITHIC_EDITOR_CONFIG=/path/to/my_config.yaml
    lithic-editor gui
    ```
    The GUI and the command line read this variable when no file is given.

A key that is not in your file keeps its default value. An unknown section or key stops the program with an error that names the key. A value of the wrong type or out of range also stops the program.

The log shows the file in use: `Configuration: <path>`.

## Sections

All lengths are in pixels at the working resolution. The working resolution is the input resolution multiplied by the upscale factor.

### resolution

| Key | Default | Meaning |
|-----|---------|---------|
| `min_line_width_px` | 6.0 | The pipeline upscales when the measured line width is below this value. |
| `min_hatch_gap_px` | 12.0 | The pipeline upscales when the measured hatch gap is below this value. |
| `max_upscale_factor` | 4 | The largest upscale factor: 2, 3 or 4. |
| `upscale_model` | espcn | The neural model: `espcn` or `fsrcnn`. |
| `dot_extent_in_line_widths` | 4.0 | A component shorter than this many line widths is a dot. Dots are not part of the hatch gap measurement. |
| `restore_ink_coverage` | 0.35 | The ink coverage at which a pixel is ink when the result returns to the input size. |

### smoothing

| Key | Default | Meaning |
|-----|---------|---------|
| `enabled` | true | Blur the image before the threshold. |
| `sigma_in_line_widths` | 0.25 | The blur sigma as a fraction of the line width. |

### threshold

| Key | Default | Meaning |
|-----|---------|---------|
| `sauvola_k` | 0.2 | The k parameter of the Sauvola threshold. |
| `window_high_dpi` | 51 | The window for a working DPI of 600 or more. |
| `window_medium_dpi` | 25 | The window for a working DPI of 300 to 599. |
| `window_low_dpi` | 15 | The window for a working DPI below 300. |

### cortex

| Key | Default | Meaning |
|-----|---------|---------|
| `reference_dpi` | 150.0 | The DPI at which the area limits are given. |
| `max_area_px` | 60 | The largest area of a cortex dot at the reference DPI. |
| `min_area_px` | 3 | The smallest area of a cortex dot at the reference DPI. |
| `max_area_floor_px` | 30 | The smallest permitted value of the upper limit. |
| `min_area_floor_px` | 2 | The smallest permitted value of the lower limit. |

The limits scale with the square of the working DPI divided by the reference DPI.

### skeleton

| Key | Default | Meaning |
|-----|---------|---------|
| `spur_length_px` | 5 | Branches shorter than this that end at a junction are removed. |
| `bridge_in_line_widths` | 1.5 | Two free ends closer than this many line widths are joined. |
| `bridge_max_hatch_gap_fraction` | 0.5 | The bridge distance is never more than this fraction of the hatch gap. |
| `bridge_min_px` | 3.0 | The smallest bridge distance. |

### ripples

| Key | Default | Meaning |
|-----|---------|---------|
| `y_tip_distance_near_300_px` | 5 | The Y-tip distance for a working DPI within 50 of 300. |
| `y_tip_distance_medium_px` | 3 | The Y-tip distance for a working DPI of 150 or more. |
| `y_tip_distance_low_px` | 2 | The Y-tip distance for a working DPI below 150. |

A junction closer than the Y-tip distance to a free end becomes a free end.

### lines

| Key | Default | Meaning |
|-----|---------|---------|
| `radius_in_line_widths` | 0.5 | Lines are rebuilt from the original ink within this fraction of the line width around the skeleton. |
| `radius_max_in_line_widths` | 0.75 | The largest rebuild radius, as a fraction of the line width. |

## Compare configurations

The evaluation harness accepts a configuration file. Run it once with the default and once with your file, then compare the two reports. See [Testing](../developer/testing.md#evaluation-harness).

```bash
python -m tools.dpi_eval.run --quick --config my_config.yaml
```
