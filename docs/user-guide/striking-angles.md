# Striking Angles

## Introduction

The ripples of a scar are arcs around the point of percussion. They record the direction of the force that made the scar. The pipeline finds the ripples before it removes them. With the striking angle option, it measures the direction of the blow for each scar.

The result is one record for each scar with enough ripples. In the GUI, each record becomes an arrow on the processed image. In the CLI, the records go to a CSV file.

## Measure the angles in the GUI

1. Keep **Add striking angle arrows** on in the Options panel. It is on by default.
2. Set the three thresholds below the option as necessary. They start at the values of the configuration file. See [the thresholds](#the-thresholds).
3. Click **Process Image**.

After processing, an arrow appears at the centre of the ripples of each scar. The arrow points in the direction of the force, away from the point of percussion, like the arrows in archaeological illustrations. The log shows the angle, the number of ripples and the spread for each scar. The log also shows each scar that has too few ripples.

The arrows are the same as the arrows you add by hand. You can move, turn, resize and delete them. See [Arrow Annotations](arrows.md). Check each arrow before you save. The saved image includes the arrows.

## The thresholds

| In the GUI | In the CLI | Default | Configuration key | Meaning |
|------------|------------|---------|-------------------|---------|
| Minimum ripples | `--min-ripples N` | 4 | `angles.min_ripples` | A scar with fewer ripples gets no angle. |
| Minimum arc depth | `--min-arc-depth PX` | 2.0 | `angles.min_arc_depth_px` | When the mean arc depth of the ripples is less than this many pixels, the direction is not certain. |
| Maximum cortex dots in a scar | `--max-cortex-dots N` | 2 | `angles.max_cortex_dots` | A face with more cortex dots is cortex, not a scar, and gets no angle. |

The GUI spin boxes and the CLI options replace the values of the [configuration file](configuration.md#angles) for one run. A loaded configuration file sets the spin boxes to its values.

## Measure the angles in the CLI

```bash
lithic-editor input.png --angles
```

The command prints one line for each scar and writes `input_angles.csv` to the output directory. See [the CSV file](#the-csv-file).

The thresholds are options. Each option replaces the value from the configuration file:

```bash
lithic-editor input.png --angles --min-ripples 3
```

## How the pipeline measures an angle

1. The structural lines enclose faces. Each face is a scar, the platform, or the space between views.
2. Each ripple belongs to the face that holds most of its pixels.
3. A face that holds more than `angles.max_cortex_dots` (default 2) cortex dots is cortex, not a scar. It is skipped. A face with fewer ripples than `angles.min_ripples` (default 4) is also skipped.
4. The orientation of each ripple is the direction of its longest axis.
5. The mean orientation of the ripples is computed in a way that treats a ripple and the same ripple turned by 180° as equal. The spread of the orientations is also computed.
6. The striking line is the line through the centre of the ripples, perpendicular to the mean orientation.
7. Each ripple is an arc. Its arc depth is the distance from the straight line between its two ends to the point of the ripple farthest from that line. The ripples curve in the direction of the force, away from the point of percussion. The pipeline takes the direction of the mean arc depth along the striking line. When the mean arc depth is smaller than `angles.min_arc_depth_px` (default 2 px), the ripples are too straight to tell the direction. The record then says that the direction is not certain. The arrow is still added.

Both thresholds are in the [configuration file](configuration.md#angles).

## The angle frame

The angle is in the frame that PyLithics uses for `arrow_angle`. It is the direction the arrowhead points, clockwise, with down at 0°. An arrow at the top of the drawing that points down into the piece is at 0°. An arrow at the right edge that points left into the piece is at 90°.

| Direction of the arrowhead | Angle |
|----------------------------|-------|
| Down | 0° |
| Left | 90° |
| Up | 180° |
| Right | 270° |

An arrow that Lithic Editor adds and PyLithics then detects gets the same angle in both tools.

## The CSV file

`<stem>_angles.csv` has one row for each scar with enough ripples. `<stem>` is the file name of the input without its extension. The file has only the header row when no scar has enough ripples.

| Column | Description |
|--------|-------------|
| `scar` | The number of the scar, from 1. The numbers are in the order of the faces, not in a spatial order. |
| `x`, `y` | The centre of the ripples, in pixels of the result. The origin is the top left corner. |
| `compass` | The direction of the force, away from the point of percussion. See [The angle frame](#the-angle-frame). |
| `ripples` | The number of ripples in the scar. |
| `spread` | 0 when all ripples are parallel. A value near 1 means the orientations are spread over all directions. Then the striking line has little meaning. |
| `direction_certain` | `True` when the ripples curve enough to tell the direction. `False` when the angle can be 180° wrong. |
| `extent` | The square root of the area of the box around the ripples, in pixels. A measure of the size of the scar. The GUI sizes the arrow from it: one fifth of the extent, never below the minimum detectable size and never above four times it. |

The Python API returns the same records as a list of dictionaries in `striking_angles`. See [Python API](../api-reference/python-api.md#return-value).

## Debug image

With the debug images on, the pipeline writes `6a_<name>_striking_angles.png`. It shows the ripple image with the striking line, the centre and the direction of each scar. Yellow arrows are certain. Orange arrows are not certain. The number next to each arrow is the scar number and the angle.

With the debug images on, the pipeline also writes `10_<name>_striking_arrows.png`: the cleaned result with a filled arrow for each scar, as the GUI places them.

## Limits

- A ripple that touches a contour at both ends has no free end. The pipeline does not see it as a ripple, so it does not count for the angle.
- Straight, parallel hatch lines give a striking line but no direction. The record says that the direction is not certain.
- Two scars that share one face, because a contour is missing, are measured as one scar.
- A face that touches the image border is the outside of the drawing. Ripples in it are not measured.
- With **Keep the cortex stipple** off, the pipeline has no cortex mask. A cortex face with large stipple can then get an arrow.

## Next steps

Check the arrows, then continue to [Output](output.md) to save the result.
