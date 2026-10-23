# Command Line

The `lithic-editor` command processes images without the GUI. Every action is a flag. The drawing to process is the one positional argument. This page shows the usual commands. For all options, see the [CLI Reference](../api-reference/cli-reference.md).

## Process one image

```bash
lithic-editor input.png
```

The result is written to `image_debug/input_cleaned.png`. The DPI tag is the input DPI.

## Set the output directory

```bash
lithic-editor input.png -o results/
```

## Write the debug images

```bash
lithic-editor input.png --debug
```

See [Debug images](processing.md#debug-images) for the file names.

## Allow upscaling

```bash
lithic-editor scan_150dpi.png --auto-upscale --default-dpi 150
```

`--auto-upscale` allows upscaling for processing when the lines are too thin or too close together. The factor is measured from the drawing. `--default-dpi` gives the DPI to assume when the file has no DPI tag. Add `--upscale-model fsrcnn` to use the FSRCNN model instead of ESPCN.

## Keep the upscaled size with a scale image

```bash
lithic-editor input.png --auto-upscale --keep-upscaled --scale-image scale.png
```

The result keeps the working size. The scale image is written as `input_scale.png`, scaled by the same factor. See [Output](output.md).

## Measure the striking angles

```bash
lithic-editor input.png --angles
```

The command measures the striking angle of each scar from its ripples. It prints one line for each scar and writes `input_angles.csv`. See [Striking angles](striking-angles.md).

The three thresholds of the measurement are options. Each option replaces the value from the configuration file:

```bash
lithic-editor input.png --angles --min-ripples 3 --min-arc-depth 1.5 --max-cortex-dots 0
```

## Process the cortex stipple as lines

```bash
lithic-editor input.png --no-preserve-cortex
```

## Process many images

```bash
for file in *.png; do
    lithic-editor "$file" -o processed/ -q
done
```

`-q` stops the processing messages.

## Other flags

| Command | Function |
|---------|----------|
| `lithic-editor --gui` | Starts the GUI. |
| `lithic-editor --help` | Shows the full help. `lithic-editor --help-api` shows the Python API help. |
| `lithic-editor --docs` | Opens the documentation in the browser. `--docs --offline` serves it on this computer. |
| `lithic-editor --version` | Shows the version. |
