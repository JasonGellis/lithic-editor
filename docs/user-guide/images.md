# Lithic Illustrations - Image Types and Quality

## Introduction

This section covers image types, formats, and quality requirements for optimal processing with Lithic Editor and Annotator. Understanding these requirements will ensure the best results when removing ripple lines from your lithic drawings.

## Supported Image Formats

### Recommended Formats
- **PNG**: Lossless compression with transparency support. Best for line drawings and technical illustrations
- **TIFF/TIF**: Professional quality, uncompressed format ideal for archival purposes
- **JPEG/JPG**: Widely compatible with smaller file sizes, but may introduce compression artifacts
- **BMP**: Simple uncompressed format, larger file sizes

## Image Quality Requirements

<div style="float: right; margin-left: 20px; margin-bottom: 20px; max-width: 350px;">
  <img src="/assets/images/lithic_300dpi.png" alt="Example lithic flake at 300 DPI" style="width: 100%; border: 1px solid #ddd; padding: 10px; background: white;">
  <p style="font-size: 0.9em; font-style: italic; text-align: center; margin-top: 8px; color: #666;">
    Example of a high-quality lithic flake illustration at 300 DPI showing ripple lines, cortex stippling, and clear structural boundaries suitable for processing
  </p>
</div>

For optimal processing results, your images should have:

- **High contrast**: Black lines on white background work best
- **Resolution**: Minimum 300 DPI (dots per inch) recommended
- **Clean lines**: Continuous, unbroken strokes for structural elements
- **Visible ripples**: Ripple lines should extend from one edge of the scar but not meet the opposite edge
- **Minimal noise**: Free from scanning artifacts or background texture

## Resolution and DPI

### Automatic DPI Detection
Lithic Editor automatically:

- Reads DPI metadata from image files
- Prompts for manual DPI input when metadata is missing
- Offers neural network upscaling for images below 300 DPI

### DPI Requirements
- **Optimal**: 300 DPI or higher
- **Minimum**: 150 DPI (will trigger automatic upscaling)
- **Maximum tested**: 600 DPI

## Low-Resolution Image Enhancement

### Neural Network Upscaling
For images below 300 DPI, Lithic Editor provides options for automatic enhancement using deep learning models:

**ESPCN (Efficient Sub-Pixel CNN)**

- Faster processing speed
- Suitable for most lithic drawings
- 2x upscaling factor

**FSRCNN (Fast Super-Resolution CNN)**

- Higher quality results
- Better edge preservation
- Ideal for complex illustrations

For technical details about these super-resolution models, see [OpenCV Super Resolution Tutorial](https://learnopencv.com/super-resolution-in-opencv/#sec3).

## Image Preparation Best Practices

### Before Processing
1. **Scan settings**: Use black and white or grayscale mode at 300+ DPI
2. **File format**: Save as PNG or TIFF to avoid compression artifacts
3. **Orientation**: Ensure drawings are properly aligned
4. **Cropping**: Remove unnecessary borders or scale bars

### Common Issues to Avoid
Lithic Editor and Annotator provides options for cleaning and enhancing images.
However, for the best and quickest experience try to avoid images with:

- **Low contrast**: Gray lines on off-white backgrounds reduce accuracy
- **Broken lines**: Gaps in structural elements may be misidentified
- **Text overlays**: Remove labels or annotations before processing
- **Multiple artifacts**: Process one lithic illustration per image

## Special Considerations

### Cortex Preservation
Lithic Editor intelligently handles cortex stippling:
- Automatically detects and preserves stippled areas
- Maintains archaeological accuracy
- Can be toggled on/off based on your needs

### Complex Illustrations
For drawings with intricate details:
- Enable debug mode to review processing steps
- Use the brush tool for manual pre-processing
- Consider processing in sections for very large images

## Next Steps

Once you understand image requirements and have prepared your lithic drawings, continue to [Processing Images](processing.md) to learn how to remove ripple lines and clean your illustrations.