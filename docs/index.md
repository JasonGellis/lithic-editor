# Lithic Editor and Annotator

<div class="hero-section">
  <h2>Lithic Editor and Annotator</h2>
  <p>Removes ripple lines from scanned lithic drawings, keeps contours and cortex, and adds direction arrows</p>
</div>

!!! success "Key Features"
    - ![](assets/images/lithic_tool.svg){: style="width:24px; height:24px; vertical-align:middle; margin-right:8px"}**Ripple removal** - Removes hatch lines and keeps scar contours, outlines and cortex stipple
    - ![](assets/images/arrow.svg){: style="width:24px; height:24px; transform:rotate(-45deg); vertical-align:middle; margin-right:8px; filter:brightness(0)"}**Arrow annotation** - Adds direction arrows that show the striking direction
    - ![](assets/images/article.svg){: style="width:24px; height:24px; vertical-align:middle; margin-right:8px; filter:brightness(0)"}**Upscaling for thin lines** - Upscales a drawing before processing when its lines are too thin or too close
    - ![](assets/images/smile_face.svg){: style="width:24px; height:24px; vertical-align:middle; margin-right:8px; filter:brightness(0)"}**GUI and command line** - Processes drawings from a window or from a terminal
    - ![](assets/images/api.svg){: style="width:24px; height:24px; vertical-align:middle; margin-right:8px; filter:brightness(0)"}
    **Python API** - Processes many drawings from a script

## What is Lithic Editor?

Lithic Editor and Annotator prepares scanned drawings of stone tools for analysis and publication. The input is a scanned line drawing. The output is a black-on-white drawing without ripple lines.

The software does these tasks:

- **Ripple removal**: The software finds the hatch lines that show scar ripples and removes them. It keeps scar contours, outlines and cortex stipple.

- **Cortex preservation**: The software separates cortex stipple from lines by the area of each mark. It adds the stipple back to the result.

- **Upscaling**: The software measures the line width and the hatch gap in pixels. If the lines are too thin or too close, it upscales the drawing before processing. It uses a neural model (ESPCN or FSRCNN) that is included in the package. By default, the result has the same pixel size and DPI as the input.

- **Arrow annotation**: The GUI has tools to add, move, rotate, resize and color direction arrows.

- **Three interfaces**: Use the GUI, the `lithic-editor` command, or the Python API.

## Visual Example

<div style="display: flex; flex-direction: row; gap: 30px; align-items: flex-start; margin: 20px 0;">
    <div style="flex: 1; text-align: center;">
      <h3>Before Processing</h3>
      <p>Input drawing with ripple lines and cortex stipple</p>
      <img src="assets/images/lithic_300dpi.png" alt="Before processing: lithic drawing with ripple lines and cortex stipple" style="max-width: 100%; height: auto;">
    </div>
    <div style="flex: 1; text-align: center;">
      <h3>Ripple Removal</h3>
      <p>Contours and cortex without ripple lines</p>
      <img src="assets/images/lithic_300dpi_processed.png" alt="After processing: contours and cortex without ripple lines" style="max-width: 100%; height: auto;">
    </div>
    <div style="flex: 1; text-align: center;">
      <h3>Arrow Annotation</h3>
      <p>Direction arrows show the striking direction</p>
      <img src="assets/images/lithic_300dpi_annotation.png" alt="After annotation: direction arrows replace the ripple lines" style="max-width: 100%; height: auto;">
    </div>
</div>

## Who is this for?

- **Archaeologists** who work with lithic drawings
- **Researchers** who prepare drawings for publication
- **Museum curators** who prepare artifact records
- **Students** who learn archaeological illustration

## Next Steps

1. Install the software. See the [installation guide](getting-started/installation.md).
2. Process a first drawing. See the [user guide](user-guide/overview.md).
