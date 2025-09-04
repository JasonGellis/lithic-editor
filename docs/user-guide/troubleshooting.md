## Troubleshooting Guide

### Processing Issues

| Problem | Solution |
|---------|----------|
| Structural lines removed | Adjust algorithm sensitivity |
| Ripples not fully removed | Check image contrast |
| Processing too slow | Reduce image size |
| Memory errors | Close other applications |

### Annotation Issues

| Problem | Solution |
|---------|----------|
| Arrows too small | Increase DPI setting |
| Can't select arrow | Click closer to arrow center |
| Arrows disappear | Check arrow color vs background |

## Troubleshooting

### Common Issues

??? problem "Structural lines are removed"
    **Causes:**
    - Lines too similar to ripple pattern
    - Incorrect threshold settings

    **Solutions:**
    - Increase structural preservation setting
    - Manually edit before processing
    - Use debug mode to identify issue

??? problem "Ripples not fully removed"
    **Causes:**
    - Inconsistent ripple pattern
    - Poor image quality
    - Ripples too thick

    **Solutions:**
    - Improve scan quality
    - Pre-process to enhance contrast
    - Adjust sensitivity settings

??? problem "Processing takes too long"
    **Causes:**
    - Image too large
    - Insufficient memory
    - Complex line patterns

    **Solutions:**
    - Resize image to 3000px max
    - Close other applications
    - Use batch processing overnight

### Image Preparation Tips

1. **Scanning**
   - Use 300+ DPI
   - Black and white mode
   - Clean scanner glass

2. **Editing**
   - Remove text and scales
   - Fill gaps in lines
   - Increase contrast

3. **Format**
   - Save as PNG
   - Use lossless compression
   - Preserve metadata


## Performance Optimization

### Memory Management
- Process images under 4000×4000 pixels
- Close unnecessary applications
- Use 64-bit Python

### Speed Improvements
- Resize large images first
- Process in batches overnight
- Use SSD for temp files
