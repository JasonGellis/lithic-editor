# API Reference

The Lithic Editor has two programmatic interfaces. Both interfaces use the same processing pipeline.

## Interfaces

### ![](../assets/images/api.svg){: style="width:24px; height:24px; vertical-align:text-bottom; margin-right:8px"} [Python API](python-api.md)

Call `process_lithic_drawing` from Python. The function reads a file path or a numpy array. It returns the cleaned drawing as a numpy array.

```python
from lithic_editor.processing import process_lithic_drawing

cleaned = process_lithic_drawing("drawing.png")
```

### ![](../assets/images/laptop_mac.svg){: style="width:24px; height:24px; vertical-align:text-bottom; margin-right:8px"} [Command Line Interface](cli-reference.md)

Run `lithic-editor process` from a shell. The command writes the cleaned drawing to a directory as `<stem>_cleaned.png`.

```bash
lithic-editor process drawing.png --output results/
```

## What the pipeline does

The pipeline does these steps for each drawing:

1. Read the image as grayscale. The DPI comes from the file tag or from you. The pipeline does not guess a DPI.
2. Measure the line width and the hatch gap in pixels.
3. Upscale for processing only when you permit it and the lines are too thin or too close. The factor is 2, 3 or 4. The pipeline never downscales.
4. Remove the ripple lines. Keep the scar contours, the outlines and the cortex stipple.
5. Return a black-on-white drawing at the input pixel size and DPI. You can keep the upscaled size instead.

## Which interface to use

| Task | Interface |
|------|-----------|
| Clean one drawing and add arrows | GUI: `lithic-editor gui` |
| Clean many drawings from a shell script or a Makefile | [CLI](cli-reference.md) |
| Call the pipeline from Python code, or process a numpy array | [Python API](python-api.md) |

## Next steps

- [Python API](python-api.md): the full parameter table, return values and examples.
- [CLI Reference](cli-reference.md): each command, each flag, output files and exit codes.
- [User Guide](../user-guide/processing.md): the GUI procedure.
