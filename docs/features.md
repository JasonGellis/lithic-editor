# Features

## Image Processing Pipeline

### Processing Workflow

<div style="display: flex; gap: 30px; align-items: flex-start;">

<div style="flex: 1; min-width: 45%;">

<h4>Processing Steps</h4>

<h5>Input & Preprocessing</h5>

<p><strong>1. Image Loading and Metadata Extraction</strong><br>
Reads image files and extracts embedded DPI information from EXIF metadata. Supports PNG, JPEG, TIFF, and BMP formats.</p>

<p><strong>2. Grayscale Conversion</strong><br>
Converts color images to 8-bit grayscale using luminance-preserving algorithms for consistent processing.</p>

<p><strong>3. Neural Network Upscaling (Optional)</strong><br>
Detects DPI from image metadata or prompts user for DPI selection. Upscales low-DPI images using ESPCN or FSRCNN models while maintaining aspect ratio and line quality to improve detail preservation.</p>

<p><strong>4. Binary Thresholding</strong><br>
Converts to grayscale if needed and employs Otsu's adaptive thresholding to create binary image, separating foreground lines from background.</p>

<p><strong>5. DPI-Adaptive Cortex Separation</strong><br>
Separates cortex stippling from structural lines using connected component analysis with DPI-scaled minimum/maximum thresholds. Filters noise pixels while preserving legitimate cortex stippling before skeletonization.</p>

<p><strong>6. Morphological Operations</strong><br>
Applies targeted morphological transformations exclusively to structural elements:</p>
<ul>
<li>Dilation strengthens thin lines</li>
<li>Closing bridges small gaps</li>
<li>Opening smooths irregular edges</li>
</ul>

<h5>Structural Analysis & Ripple Detection</h5>

<p><strong>7. Skeletonization</strong><br>
Reduces lines to single-pixel width using morphological thinning while preserving connectivity and topology to create network representation.</p>

<p><strong>8. Junction and Endpoint Detection</strong><br>
Analyzes skeleton connectivity to identify line terminations (endpoints) and intersections (junctions) through neighbor counting.</p>

<p><strong>9. Line Segmentation</strong><br>
Detects individual line segments and calculates orientation and length, creating discrete segments for individual analysis.</p>

<p><strong>10. Graph Construction</strong><br>
Builds connectivity graph using NetworkX with segments as edges and junctions/endpoints as nodes.</p>

<p><strong>11. Y-tip Junction Conversion</strong><br>
Converts junctions within DPI-scaled threshold distance (2-8 pixels) of endpoints to endpoints, eliminating Y-tip artifacts while preserving structural integrity.</p>

<p><strong>12. Ripple Identification</strong><br>
Identifies parallel line patterns and analyzes spacing consistency to classify segments as ripple or structural elements.</p>

<p><strong>13. Selective Removal</strong><br>
Removes identified ripple lines while preserving structural boundaries to maintain artifact integrity.</p>

<h5>Output Generation</h5>

<p><strong>14. DPI-Adaptive Thickness Reconstruction</strong><br>
Applies controlled dilation with DPI-scaled parameters (1-6 pixels) to restore original line thickness while preventing over-thickening at low resolutions.</p>

<p><strong>15. Quality Enhancement</strong><br>
Implements anti-aliasing and smoothing algorithms to produce publication-quality output.</p>

<p><strong>16. Final Assembly</strong><br>
Combines cleaned structural lines with preserved cortex and refines endpoint decisions after cleaning to create final archaeologically accurate result.</p>

<p><strong>17. Final Output</strong><br>
Produces cleaned image with proper contrast orientation and preserved metadata for publication use.</p>

</div>

<div style="flex: 1; min-width: 45%;">

```mermaid
flowchart TB
    subgraph section1 ["Input & Preprocessing"]
        direction TB
        subgraph row1a [" "]
            direction LR
            A["Load Image"] --> B["Extract Metadata"]
            B --> C["Convert to Grayscale"]
            C --> D{"`DPI < 300?`"}
            D -->|Yes| E["Neural Network<br/>Upscaling"]
            D -->|No| F["Binary Threshold"]
            E --> F
        end

        subgraph row1b [" "]
            direction LR
            G["Separate Cortex"] --> H["Morphological<br/>Operations"]
        end

        F --> G
    end

    subgraph section2 ["Structural Analysis & Ripple Detection"]
        direction TB
        subgraph row2a [" "]
            direction LR
            I["Skeletonization"] --> J["Junction/Endpoint<br/>Detection"]
            J --> K["Line Segmentation"]
        end

        subgraph row2b [" "]
            direction LR
            L["Graph Construction"] --> M["Ripple<br/>Identification"]
            M --> N["Y-tip Junction<br/>Conversion"]
            N --> O["Structural Mask<br/>Creation"]
        end

        K --> L
    end

    subgraph section3 ["Output Generation"]
        direction LR
        P["DPI-Adaptive<br/>Thickness Reconstruction"] --> Q["Quality<br/>Enhancement"]
        Q --> R["Cortex<br/>Restoration"]
        R --> S["Final Output"]
    end

    H --> I
    O --> P

    style D fill:#f9f,stroke:#333,stroke-width:2px
    style E fill:#bbf,stroke:#333,stroke-width:2px
    style section1 fill:#f8f9fa,stroke:#dee2e6,stroke-width:2px
    style section2 fill:#fff3cd,stroke:#ffeaa7,stroke-width:2px
    style section3 fill:#d1ecf1,stroke:#bee5eb,stroke-width:2px
    style row1a fill:transparent,stroke:none
    style row1b fill:transparent,stroke:none
    style row2a fill:transparent,stroke:none
    style row2b fill:transparent,stroke:none
```

</div>
