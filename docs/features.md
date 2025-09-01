# Features

## Image Processing

### Neural Network Upscaling
Automatically enhance low-resolution images using state-of-the-art neural networks:

- **ESPCN** (Efficient Sub-Pixel CNN): Fast, efficient upscaling
- **FSRCNN** (Fast Super-Resolution CNN): Higher quality results
- **Automatic DPI detection** from image metadata
- **Interactive dialogs** for user control when metadata is missing
- **300 DPI target** for optimal processing quality

Learn more about these models: [OpenCV Super Resolution Tutorial](https://learnopencv.com/super-resolution-in-opencv/#sec3)

### Intelligent Ripple Removal
The core feature uses advanced graph-based analysis to identify and remove hatching/ripple lines while preserving structural features. The algorithm:

- Analyzes line patterns using skeletonization
- Identifies ripple lines through pattern recognition
- Preserves structural elements and edges
- Maintains image quality and resolution

### Processing Visualization
View and save step-by-step processing stages to understand how the algorithm works and to troubleshoot problematic images:

- Neural network upscaling (if needed)
- Original image analysis
- Skeletonization process
- Endpoint and junction detection
- Line segment labeling
- Ripple identification
- Structural cleaning
- Endpoint filtering
- Final cleaned result

### Manual Editing Tools
Fine-tune your images before processing:

- Brush tools for cleanup
- Adjustable brush sizes
- Real-time preview

## Annotation Tools

### Directional Arrows
Add professional-quality arrows to indicate:

- Force direction on lithic artifacts
- Flake scar patterns
- Manufacturing techniques
- Impact points

### Arrow Customization
Complete control over arrow appearance:

- **Size**: Adjust arrow dimensions
- **Rotation**: Orient to any angle
- **Color**: Choose from full color palette
- **Position**: Precise placement on image

### Arrow Controls

=== "Windows/Linux"
    - **Move**: Click and drag
    - **Rotate**: Shift + drag
    - **Resize**: Alt + drag
    - **Delete**: Select and press Delete

=== "macOS"
    - **Move**: Click and drag
    - **Rotate**: Shift + drag
    - **Resize**: Option + drag
    - **Delete**: Select and press Delete

## Technical Features

### DPI Preservation
- Maintains original image resolution
- Preserves DPI metadata
- Ensures publication-quality output
- Supports high-resolution displays

### Multiple Output Formats

| Format | Features | Best For |
|--------|----------|----------|
| PNG | Lossless, transparency support | Web, presentations |
| JPEG | Compressed, wide compatibility | Publications, sharing |
| TIFF | Uncompressed, professional | Archival, printing |

### Command-Line Processing
Process images from the command line:

- Single image processing
- Consistent processing parameters
- Debug output options
- Error handling

### Cross-Platform Support
Works seamlessly on:

- ✅ Windows 10/11
- ✅ macOS 10.15+
- ✅ Linux (Ubuntu, Fedora, etc.)

## Advanced Features

### Debug Mode
Save intermediate processing steps for:

- Algorithm verification
- Quality assurance
- Research purposes
- Teaching and training

### API Integration
Integrate Lithic Editor into your workflow:

```python
from lithic_editor.processing import process_lithic_drawing

# Process with custom parameters
result = process_lithic_drawing(
    image_path="artifact.png",
    save_debug=True,
    output_dpi=300
)
```

### Command-Line Interface
Full functionality from the terminal:

```bash
# Process with options
lithic-editor process image.png --output results/ --debug

# Process multiple files with bash
for file in *.png; do
    lithic-editor process "$file" --quiet
done
```