# Command Line

The `lithic-editor` command processes images without the GUI. This page shows the usual commands. For all options, see the [CLI Reference](../api-reference/cli-reference.md).

## Process one image

```bash
lithic-editor process input.png
```

The result is written to `image_debug/input_cleaned.png`. The DPI tag is the input DPI.

## Set the output directory

```bash
lithic-editor process input.png -o results/
```

## Write the debug images

```bash
lithic-editor process input.png --debug
```

See [Debug images](processing.md#debug-images) for the file names.

## Allow upscaling

```bash
lithic-editor process scan_150dpi.png --auto-upscale --default-dpi 150
```

`--auto-upscale` allows upscaling for processing when the lines are too thin or too close together. The factor is measured from the drawing. `--default-dpi` gives the DPI to assume when the file has no DPI tag. Add `--upscale-model fsrcnn` to use the FSRCNN model instead of ESPCN.

## Keep the upscaled size with a scale image

```bash
lithic-editor process input.png --auto-upscale --keep-upscaled --scale-image scale.png
```

The result keeps the working size. The scale image is written as `input_scale.png`, scaled by the same factor. See [Output](output.md).

## Process the cortex stipple as lines

```bash
lithic-editor process input.png --no-preserve-cortex
```

## Process many images

```bash
for file in *.png; do
    lithic-editor process "$file" -o processed/ -q
done
```

`-q` stops the processing messages.

## Other commands

| Command | Function |
|---------|----------|
| `lithic-editor gui` | Starts the GUI. |
| `lithic-editor help` | Shows the help. `lithic-editor help api` shows the Python API help. |
| `lithic-editor docs` | Opens the documentation in the browser. `--offline` serves it locally. |
| `lithic-editor --version` | Shows the version. |
