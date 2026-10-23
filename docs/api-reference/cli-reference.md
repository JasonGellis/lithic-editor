# CLI Reference

This page describes each flag of the `lithic-editor` command. Every action is a flag. The drawing to process is the one positional argument.

## Synopsis

```bash
lithic-editor [--help] [--help-api] [--version] [--gui] [--docs [--offline]]
lithic-editor INPUT [options]
```

`lithic-editor` with no arguments shows the full help and exits with code 0.

## Actions

| Flag | Description |
|------|-------------|
| `-h`, `--help` | Show the full help and exit. |
| `--help-api` | Show the help for the Python API and exit. |
| `--version` | Show the version and exit. |
| `--gui` | Start the graphical interface. The command returns the exit code of the interface. If the interface does not start, the command prints a message and exits with code 1. |
| `--docs` | Open the online documentation in the web browser. |
| `--docs --offline` | Serve the documentation that comes with the package at `http://127.0.0.1:8000` and open it in the web browser. Press Ctrl+C to stop the server. The command exits with code 1 when the documentation files are not found or when port 8000 is in use. |

## Process a drawing

Remove the ripple lines from one drawing and write the result to a directory.

```bash
lithic-editor INPUT [-o DIR] [--debug] [-q] [--auto-upscale] [--default-dpi DPI]
              [--upscale-model {espcn,fsrcnn}] [--config PATH] [--keep-upscaled]
              [--scale-image PATH] [--angles] [--min-ripples N] [--min-arc-depth PX]
              [--max-cortex-dots N] [--no-preserve-cortex]
```

**Argument**

| Argument | Description |
|----------|-------------|
| `INPUT` | The path of the drawing. Permitted extensions: `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp`. |

**Options**

| Option | Default | Description |
|--------|---------|-------------|
| `-o DIR`, `--output DIR` | `image_debug` | The output directory. The command makes the directory when it does not exist. |
| `--debug` | off | Write the debug images to the output directory. |
| `-q`, `--quiet` | off | Do not print the processing messages. Error messages are still printed. |
| `--auto-upscale` | off | Permit upscaling for processing when the lines are too thin or too close. The factor is 2, 3 or 4, measured from the drawing. |
| `--default-dpi DPI` | none | The DPI to use when the input file has no DPI tag. |
| `--upscale-model {espcn,fsrcnn}` | `espcn` | The neural model for upscaling. |
| `--keep-upscaled` | off | Keep the result at the upscaled working size. The DPI tag is the input DPI multiplied by the factor. |
| `--config PATH` | the shipped file | A configuration file. See [Configuration](../user-guide/configuration.md). |
| `--scale-image PATH` | none | The scale bar image scanned with the drawing. See [Output files](#output-files). |
| `--angles` | off | Measure the striking angle of each scar from its ripples. See [Striking angles](../user-guide/striking-angles.md). |
| `--min-ripples N` | 4 | A scar with fewer ripples than N gets no angle. |
| `--min-arc-depth PX` | 2.0 | When the mean arc depth of the ripples of a scar is less than PX pixels, the direction is marked uncertain. |
| `--max-cortex-dots N` | 2 | A face with more than N cortex dots is cortex and gets no angle. |
| `--no-preserve-cortex` | off | Process the cortex stipple as lines. By default the pipeline keeps the stipple. |

The pipeline does not guess a DPI. When the file has no DPI tag and you give no `--default-dpi`, the pipeline uses fixed size thresholds. The result then has no DPI tag.

Without `--auto-upscale`, the pipeline processes the drawing at its input size. Without `--keep-upscaled`, the pipeline returns the result at the input size and DPI, also when it upscaled for processing.

### Output files

The command writes these files to the output directory. `<stem>` is the file name of the input without its extension.

| File | When written | Content |
|------|--------------|---------|
| `<stem>_cleaned.png` | always | The cleaned drawing, black lines on white. The DPI tag is the input DPI. With `--keep-upscaled`, the DPI tag is the input DPI multiplied by the factor. No DPI tag is written when the input has none. |
| `<stem>_scale.png` | with `--scale-image` and `--keep-upscaled`, when the factor is more than 1 | The scale bar image, scaled by the same factor, with the same DPI tag as the result. |
| `<stem>_angles.csv` | with `--angles` | One row for each scar with enough ripples. See [Striking angles](../user-guide/striking-angles.md#the-csv-file). |
| debug images | with `--debug` | One PNG file for each processing step. See the [debug image list](python-api.md#debug-images). |

With `--scale-image` and no `--keep-upscaled`, the command does not write a scale file. The result is at the input size, so the scale bar image is correct as scanned.

**Messages**

Without `--quiet`, the command prints the input path, the output directory, the pipeline progress and the path of the result. When the result is larger than the input, the command prints the factor. When the result is larger and no scale image was given, the command prints a reminder. Scale the scale bar image by the same factor before you measure.

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | The command completed. |
| 1 | The input file does not exist, the extension is not permitted, the processing failed, the interface did not start, or the user stopped the command. |
| 2 | The command line is not valid. The argument parser prints the usage. |

## Examples

```bash
# Clean one drawing. The result is image_debug/drawing_cleaned.png.
lithic-editor drawing.png

# Write the result to a different directory.
lithic-editor drawing.png --output results/

# Write the debug images.
lithic-editor drawing.png --debug

# Print no processing messages.
lithic-editor drawing.png --quiet

# Permit upscaling for a thin-lined scan with no DPI tag.
lithic-editor scan_150dpi.png --auto-upscale --default-dpi 150

# Permit upscaling with the FSRCNN model.
lithic-editor drawing.png --auto-upscale --upscale-model fsrcnn

# Keep the upscaled size and scale the scale bar by the same factor.
lithic-editor drawing.png --auto-upscale --keep-upscaled --scale-image scale_bar.png

# Measure the striking angle of each scar. The angles are written to drawing_angles.csv.
lithic-editor drawing.png --angles

# Measure the angle of each scar with three or more ripples.
lithic-editor drawing.png --angles --min-ripples 3

# Process the cortex stipple as lines.
lithic-editor drawing.png --no-preserve-cortex

# Start the graphical interface.
lithic-editor --gui

# Open the documentation without a network connection.
lithic-editor --docs --offline
```

## Batch processing

### Shell loop

Process each PNG file in a directory. Write all results to one directory.

```bash
for file in drawings/*.png; do
    lithic-editor "$file" --output processed/ --quiet
done
```

Record the files that failed.

```bash
for file in drawings/*.png; do
    if lithic-editor "$file" --output processed/ --quiet; then
        echo "$file" >> success.log
    else
        echo "$file" >> failed.log
    fi
done
```

### find and xargs

Process the PNG files in a directory tree. Run four processes at the same time.

```bash
find drawings/ -name "*.png" -print0 | \
    xargs -0 -n 1 -P 4 -I {} lithic-editor {} --output processed/ --quiet
```

### GNU parallel

```bash
parallel -j 4 lithic-editor {} --output processed/ --quiet ::: drawings/*.png
```

### Makefile

Make a `<stem>_cleaned.png` file for each PNG file in `drawings/`. `make` processes only the files with no result or with a newer input.

```makefile
INPUT_DIR = drawings
OUTPUT_DIR = processed
IMAGES = $(wildcard $(INPUT_DIR)/*.png)
OUTPUTS = $(patsubst $(INPUT_DIR)/%.png,$(OUTPUT_DIR)/%_cleaned.png,$(IMAGES))

all: $(OUTPUTS)

$(OUTPUT_DIR)/%_cleaned.png: $(INPUT_DIR)/%.png
	lithic-editor $< --output $(OUTPUT_DIR) --quiet

clean:
	rm -rf $(OUTPUT_DIR)

.PHONY: all clean
```

!!! note
    A Makefile recipe line must start with a tab character, not with spaces.
