# Processing Images

This page tells you how to process one image in the GUI. It also describes the pipeline stages and the debug images. For the command line, see [Command Line](CLI.md). For the Python API, see the [Python API](../api-reference/python-api.md).

## Procedure

### Load the image

1. Click **Load Image**.
2. Select a PNG, JPEG, BMP or TIFF file.

The application crops the image to its content and shows it in the **Input Image** panel. The **Processing Log** shows the DPI, the format and the size. The **DPI** section shows the DPI.

If the image has no DPI, select **No DPI value** or **Set the DPI** in the DPI section. This choice sets the DPI tag of the saved file. See [Output](output.md).

### Edit the input (optional)

Use the brush to remove marks or to close gaps before processing.

1. Click **Activate Brush**.
2. Select **White** to erase or **Black** to draw.
3. Set the **Size**.
4. Drag on the **Input Image** panel.
5. Click **Clear Brush** to remove all brush marks.

The brush does not change the file on disk. Processing uses the edited image.

### Set the options

- Set **Show and save the debug images** to examine each stage. See [Debug images](#debug-images).
- Keep **Keep the cortex stipple** on for drawings with cortex stipple. Turn it off to process the stipple as lines.
- Set **Keep the upscaled size** and load a scale image if you want the result at the upscaled working size. See [Output](output.md).

### Process the image

1. Click **Process Image**.
2. If the image has no DPI, the **Specify Image DPI** dialog opens. Click **72**, **96**, **150** or **200**. Or enter a value from 50 to 2400 and click **Use Custom**. Click **Cancel** to process at the original size without a DPI.
3. The application measures the line width and the hatch gap. If the line width is below 6 px or the hatch gap is below 12 px, the **Upscale Image** dialog opens. It shows the measured values and the upscale factor.
    - Select a model: **ESPCN** (recommended) or **FSRCNN**.
    - Set or clear **Keep the upscaled size**. The text below it tells you the size and the DPI of the result. The choice also sets the checkbox in the Options panel. See [Output](output.md).
    - Click **Upscale** to upscale for processing.
    - Click **Do not upscale** to process at the original size.
4. Wait until the status shows "Processing complete".

The buttons are not available during processing. The log shows each stage.

### Examine the result

1. Compare the **Input Image** and the **Processed Image** panels.
2. Make sure that the scar contours and the outline are complete.
3. Make sure that the ripple lines are removed.
4. If the result is not correct, examine the debug images. Then edit the input with the brush and process again.

Then continue to [Arrow Annotations](arrows.md) and [Output](output.md).

## Pipeline stages

The pipeline does these steps in this order.

1. Loads the image as grayscale. The DPI comes from the file tag or from you. The pipeline does not guess a DPI.
2. Measures the line width and the hatch gap in pixels.
3. Upscales for processing only when necessary. The factor is the smallest of 2, 3 or 4 that makes the line width at least 6 px and the hatch gap at least 12 px. The model is ESPCN (default) or FSRCNN. Both models are included in the package. The pipeline never downscales.
4. Smooths the image with a Gaussian blur. The sigma is 0.25 times the line width. This separates the tips of the hatch lines from the contours they touch.
5. Thresholds the image to a binary image with the Sauvola method. The pipeline also keeps an unsmoothed threshold for step 12.
6. Separates the cortex stipple from the structural lines by component area. The area thresholds scale with the DPI.
7. Skeletonizes the structural lines (Lee method), closes small gaps, thins again and removes short spurs.
8. Bridges free skeleton ends that are closer than about 1.5 times the line width, and never more than half a hatch gap. A contour with a small break stays closed.
9. Finds endpoints (one neighbour) and junctions (three or more neighbours). Converts junctions very near an endpoint (Y-tips) to endpoints.
10. Splits the skeleton at the endpoints and junctions into segments and builds a graph.
11. Classifies a segment that touches a free endpoint as a ripple. All other segments are structural.
12. Rebuilds the structural lines from the unsmoothed ink at a radius of half the line width. The lines keep the original pen weight. Adds the cortex back. Inverts the image to black on white.
13. Returns the result at the input pixel size and DPI. With the keep-upscaled option, returns the working size instead. See [Output](output.md).

## Debug images

Set **Show and save the debug images** to write the debug images. The GUI writes them to `image_debug/<name>/`. `<name>` is the input file name without its extension. The CLI writes them to the output directory. The images appear in this order.

| File | Content |
|------|---------|
| `0_<name>_input.png` | The input image. Written only when the image is upscaled. |
| `0a_<name>_upscaled.png` | The upscaled image. Written only when the image is upscaled. |
| `0b_<name>_upscaled_scale.png` | The upscaled scale image. Written only with a scale image and keep-upscaled. |
| `1_<name>_original_image.png` | The grayscale image at the working size. |
| `1b_<name>_smoothed.png` | The image after the Gaussian blur. |
| `1c_<name>_binary_thresholded.png` | The binary image after the Sauvola threshold. |
| `2a_<name>_structural_only.png` | The structural lines without the cortex stipple. |
| `2b_<name>_cortex_mask.png` | The cortex stipple. |
| `3_<name>_skeleton.png` | The skeleton of the structural lines. |
| `4_<name>_endpoints_junctions.png` | The endpoints and the junctions. |
| `5_<name>_labeled_segments.png` | The segments, each in a different color. |
| `6_<name>_ripple_identification.png` | Red = ripple, white = structural, green dots = junctions. |
| `7_<name>_skeleton_cleaned.png` | The skeleton without the ripple segments. |
| `7a_<name>_endpoint_filtering.png` | Gray = skeleton, white = structural, green = junctions, blue = kept endpoints, red = removed endpoints. |
| `8_<name>_final_cleaned.png` | The rebuilt lines with the cortex, black on white. |
| `9_<name>_restored.png` | The result at the input size. Written only when the image was upscaled and returned to the input size. |

!!! tip "Find the cause of a problem"
    Examine `6_<name>_ripple_identification.png` first. A red segment is removed. A white segment is kept.
