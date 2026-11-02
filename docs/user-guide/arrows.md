# Arrow Annotations

## Introduction

Arrows show a direction on the processed drawing. The arrow tools become available after processing. The saved image includes the arrows.

When **Add striking angle arrows** is on, processing adds one arrow for each scar with enough ripples. These arrows are the same as the arrows you add by hand. See [Striking angles](striking-angles.md).

## Add an arrow

1. Process the image.
2. Click **Arrow Color** and select a color. The default is black. This step is optional.
3. Select a style in **Arrow Style**. The default is Filled. This step is optional.
4. Click **Add Arrow**.

The arrow appears at the center of the processed image and is selected. It points to the right.

## Arrow styles

Lithic illustrations use several arrow styles. Each style has a meaning. The styles follow Inizan et al. (1999, fig. 52), Raczynski-Henk (2017, fig. 7) and Martingell and Saville (1988, fig. 27). The terms are in the [glossary](../glossary.md#arrow-symbols).

| Style | Shape | Meaning |
|-------|-------|---------|
| **Filled** | A filled head on a shaft. | The direction of the blow. The default. The striking angle arrows use this style. |
| **Open** | An outline head on a shaft. | The bulb imprint is not visible in the scar. Used on burin facets. |
| **Dot** | A filled arrow with a dot at the tail. | A bulb or a negative bulb is present at that end. |
| **Crossed** | A filled arrow with a bar across the shaft. | The orientation and direction of debitage. |
| **Dashed** | A filled head on a dashed shaft. | An earlier burin blow. |

To change the style of an arrow:

1. Click the arrow to select it. **Arrow Style** shows its style.
2. Select the new style in **Arrow Style**.

The style changes the shape only. The size, color, position and angle stay the same. Each part of the arrow is drawn in the arrow color and scales with the arrow size.

## Select an arrow

Click near the center of the arrow. A blue dashed circle shows the selected arrow. Click an empty area to deselect. When arrows overlap, the click selects the arrow on top.

## Move an arrow

Drag the arrow to the new position.

## Rotate an arrow

- Hold Shift and drag. The arrow turns around its center.
- Hold Shift and Ctrl (Command on macOS) and drag to turn in steps of 15 degrees.
- Press the Left or Right arrow key to turn the selected arrow 5 degrees.

## Resize an arrow

Hold Option (macOS) or Alt (Windows, Linux) and drag. Drag to the right to make the arrow larger. Drag to the left to make it smaller.

The size stays between the minimum for the image DPI and 200 px. The minimum is 50 px at 300 DPI or more. Below 300 DPI, the minimum increases in proportion. At 150 DPI, the minimum is 100 px.

## Remove arrows

- Click **Delete Arrow** or press the Delete key to remove the selected arrow.
- Click **Clear Arrows** to remove all arrows.

## Arrow size and detection

The application keeps each arrow large enough for automatic detection in measurement software.

- A new arrow gets the minimum size for the image DPI or more.
- A resize cannot make an arrow too small.
- Before a save, the application checks all arrows and enlarges the arrows that are too small. The log shows the number of changed arrows.

## Notes

- Arrows are drawn without anti-aliasing. The edges stay sharp.
- The application removes all arrows when you load a new image or process the image again. Add the arrows after the last processing run.
- Arrows are part of the saved image. You cannot edit them after the save.

## Next steps

Continue to [Output](output.md) to save the result.
