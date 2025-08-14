# Features

## Image Processing

### Intelligent Ripple Removal
The core feature of Lithic Editor uses advanced graph-based analysis to identify and remove hatching/ripple lines while preserving structural features. The algorithm:

- Analyzes line patterns using skeletonization
- Identifies ripple lines through pattern recognition
- Preserves structural elements and edges
- Maintains image quality and resolution

### Processing Visualization
View step-by-step processing stages to understand how the algorithm works:

- Original image analysis
- Skeletonization process
- Line segment detection
- Ripple identification
- Final cleaned result

### Manual Editing Tools
Fine-tune your images before processing:

- Brush tools for cleanup
- Eraser for removing artifacts
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

### Intuitive Controls

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

### Batch Processing
Process multiple images efficiently:

- Command-line batch operations
- Consistent processing parameters
- Progress tracking
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
from lithic_editor.processing import process_lithic_drawing_improved

# Process with custom parameters
result = process_lithic_drawing_improved(
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

# Batch processing
for file in *.png; do
    lithic-editor process "$file" --quiet
done
```