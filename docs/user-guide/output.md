# Output

This page describes the size, the DPI tag and the file formats of the processed image.

## Output size and DPI

By default, the result has the same pixel size and the same DPI as the input. This is also true when the pipeline upscales the image for processing. The pipeline returns the result to the input pixel grid. A scale bar scanned with the drawing keeps its meaning.

The DPI tag of the saved file follows these rules:

| Input | DPI tag of the result |
|-------|-----------------------|
| The file has a DPI tag | The input DPI |
| No DPI tag, **Set the DPI** selected | The custom DPI |
| No DPI tag, **No DPI value** selected | No DPI tag |

The application never writes a guessed DPI tag. A wrong tag gives wrong measurements later.

## Keep the upscaled size

Set **Keep the upscaled size** (in the Options panel or in the Upscale Image dialog) or `--keep-upscaled` (CLI) to keep the result at the upscaled working size. The lines are smoother and the file is larger.

With this option:

- The pixel size is the input size times the upscale factor.
- The DPI tag is the input DPI (or the custom DPI) times the upscale factor. The physical size stays the same.
- A scale image loaded with the drawing is upscaled by the same factor and saved next to the result.

If the pipeline does not upscale, this option has no effect.

## Scale image

Keep the scale bar in a separate image scanned at the same DPI as the drawing. Load it with **Load Scale Image...** (GUI) or `--scale-image PATH` (CLI).

With **Keep the upscaled size**, the application saves the scale image next to the result:

- GUI: `<name>_scale.<ext>`, in the same format as the result.
- CLI: `<stem>_scale.png` in the output directory.

Without **Keep the upscaled size**, the result has the input size and the scale image is not changed.

If you keep the upscaled size without a scale image, the log shows a warning. Scale a separate scale bar by the same factor before you measure.

## Save the result in the GUI

1. Click **Save Result**.
2. Select the format: PNG, JPEG or TIFF.
3. Enter a file name and click **Save**.

The application adds the extension when it is missing. The saved image includes the arrows. Before the save, the application makes each arrow large enough for detection. The log shows the number of changed arrows.

## Save the result in the CLI

The CLI writes the result to the output directory (`-o DIR`, default `image_debug`) as `<stem>_cleaned.png`. The DPI tag is the input DPI. No tag is written when the input has none. With `--keep-upscaled`, the tag is raised by the upscale factor.

## Debug images

Set **Show and save the debug images** (GUI) or `--debug` (CLI) to write the debug images. See [Debug images](processing.md#debug-images).
