# Glossary

The terms are in the order the pipeline uses them.

## Resolution and size

**DPI (dots per inch)**
: The number of pixels for one inch of paper. The DPI is a value in the file, not a property of the pixels. An image of 323 pixels at 75 DPI is 4.3 inches wide.

**Pixel size**
: The width and the height of the image in pixels. A change of the pixel size changes the image. A change of the DPI value does not.

**Working resolution**
: The pixel size at which the pipeline processes the image. It is the input size multiplied by the upscale factor.

**Resample**
: Make a new image with a different pixel size from an image. Upscale and downscale are the two types of resample.

**Upscale**
: Resample to more pixels. Lithic Editor upscales with the ESPCN or FSRCNN neural model by a factor of 2, 3 or 4, for processing only. The model adds probable detail between the pixels. It does not recover detail that the scanner did not record.

**Downscale**
: Resample to fewer pixels. Lithic Editor does not downscale before processing. It downscales once at the end, to return the result to the input pixel size. See *Restore*.

**Restore**
: Downscale the result to the pixel size and the DPI of the input. This is the default. The other option keeps the working size and multiplies the DPI value by the upscale factor. See [Output](user-guide/output.md).

**Anti-aliasing**
: Gray pixels along an edge. The shade of each pixel shows how much of the pixel the line covers. A scanner makes anti-aliased edges.

## Measurement

**Line width**
: The usual width of a pen stroke in pixels, measured from the drawing. It decides if the pipeline upscales, and how wide the pipeline rebuilds the lines.

**Hatch gap**
: The usual clear space between two adjacent hatch lines, in pixels. It decides if the pipeline upscales. Lines that are too near to each other merge.

**Floor**
: The smallest permitted line width (6 px) and hatch gap (12 px). When a measured value is below its floor, the pipeline upscales. See [Configuration](user-guide/configuration.md).

## Clean the image

**Smoothing**
: A small blur, with a sigma of a quarter of the line width. The thin tips of the hatch lines fall below the threshold, so they separate from the contours they touch.

**Threshold (Sauvola)**
: The step that changes the gray image to black and white. The Sauvola method selects the limit for each pixel from the brightness around it, so light and dark areas both give correct lines.

**Cortex separation**
: The step that removes the small dots of cortex stipple by area, so they are not processed as lines. The pipeline puts them back at the end.

## Find the lines

**Skeleton**
: The centre line of each stroke, one pixel wide. The pipeline builds the graph on it.

**Spur**
: A short branch of a few pixels on the skeleton, made by the thinning step. The pipeline removes spurs.

**Endpoint**
: A skeleton pixel with one neighbour: the free end of a line.

**Junction**
: A skeleton pixel with three or more neighbours: a point where lines meet.

**Bridge**
: The step that joins two free ends that are closer than about 1.5 line widths. A contour with a small break stays a closed loop.

**Y-tip**
: A junction very near an endpoint, made where the tip of a hatch line divides. The pipeline changes it to an endpoint, so the hatch line keeps a free end.

**Segment**
: A part of the skeleton between endpoints and junctions.

## Classify and rebuild

**Ripple**
: A segment that touches a free endpoint. A hatch line starts at a contour and ends in the open, so it always has one. The pipeline removes ripples.

**Structural line**
: Each other segment: an outline, a scar contour or an arris. Structural lines make closed loops, or go from junction to junction. The pipeline keeps them.

**Rebuild**
: The step that draws the structural lines again at the pen width. The pipeline keeps the original ink within half a line width of the structural skeleton.

**Restore cortex**
: The step that adds the cortex stipple back to the result.

## Evaluation

**Reference**
: The output that the harness compares each strategy with: the `native` strategy at 600 DPI. See [Testing](developer/testing.md#evaluation-harness).

**Registration**
: The alignment of two scans of one drawing by a shift and a rotation. The harness can then compare them pixel by pixel.

**Tolerance**
: The largest distance in pixels between two skeleton pixels that count as the same line. The harness uses 4 px at 600 DPI.

**Ground truth**
: A drawing cleaned by hand, used as the correct answer. There is no ground truth yet.
