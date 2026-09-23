# Troubleshooting

## Processing problems

??? failure "A scar contour or the outline is removed"
    **Cause:** the line has a free end. The pipeline classifies a segment with a free end as a ripple.

    **Solutions:**

    - Set **Show and save the debug images**. Examine `6_<name>_ripple_identification.png`. A red segment is a ripple.
    - Close the gap in the line with the black brush and process again.
    - If the lines are thin, accept the upscale when the application offers it.

??? failure "Ripple lines are not removed"
    **Cause:** the ripple line touches a contour at both ends, or the hatch lines merge.

    **Solutions:**

    - Examine `6_<name>_ripple_identification.png`. A white segment is structural.
    - Cut the ripple line from the contour with the white brush and process again.
    - Accept the upscale when the application offers it. A larger hatch gap separates the lines.
    - Scan at a higher DPI.

??? failure "Cortex stipple is removed or changed"
    **Solutions:**

    - Make sure that **Keep the cortex stipple** is on.
    - Examine `2b_<name>_cortex_mask.png` to see what the pipeline identified as stipple.
    - Make sure that the DPI is correct. The area thresholds scale with the DPI.

??? failure "The application asks for a DPI"
    **Cause:** the file has no DPI tag.

    **Solutions:**

    - Enter the DPI of the scan. Click **72**, **96**, **150**, **200** or **Use Custom**.
    - Save the scan with a DPI tag next time.
    - In the CLI, give `--default-dpi`.

??? failure "The application offers to upscale"
    **Cause:** the measured line width is below 6 px or the hatch gap is below 12 px.

    **Solutions:**

    - Click **Upscale**. The saved result keeps the input size and DPI unless **Keep the upscaled size** is on.
    - Click **Do not upscale** to process at the original size. Thin lines can break.
    - In the CLI, give `--auto-upscale`.

??? failure "Processing is slow"
    **Cause:** a large image or a large upscale factor.

    **Solutions:**

    - Crop the image to the drawing.
    - Process one drawing for each image.
    - A 4x upscale makes the working image 16 times larger. Scan at 300 to 600 DPI to avoid it.

??? failure "Processing failed"
    **Solutions:**

    - Read the last lines of the **Processing Log**.
    - Make sure that the file is a valid PNG, JPEG, BMP or TIFF image.
    - Make sure that the drawing is dark on a light background.

## Arrow problems

??? failure "The arrow cannot be selected"
    Click near the center of the arrow. The selection radius is two thirds of the arrow size.

??? failure "The arrow does not get smaller"
    The minimum arrow size depends on the DPI. Below 300 DPI the minimum is larger. The application keeps each arrow large enough for detection.

??? failure "The arrows are removed"
    The application removes all arrows when you load or process an image. Add the arrows after the last processing run.

??? failure "The arrow is not visible"
    Select a color that contrasts with the background with **Arrow Color**.

## Output problems

??? failure "The saved file has no DPI tag"
    **Cause:** the input has no DPI tag and **No DPI value** is selected.

    **Solution:** select **Set the DPI** and enter the DPI before you save.

??? failure "The scale bar does not match the result"
    **Cause:** the result was kept at the upscaled size.

    **Solutions:**

    - Load the scale image with **Load Scale Image...** before processing. The application saves it scaled by the same factor.
    - Or turn off **Keep the upscaled size**. The result then has the input size.

## Get help

Open an issue at [GitHub](https://github.com/JasonGellis/lithic-editor/issues). Include the log messages and, if possible, the image.
