## Troubleshooting Guide

### Processing Issues

| Problem | Solution |
|---------|----------|
| Structural lines removed | Check DPI detection; algorithm auto-adapts thresholds |
| Ripples not fully removed | Ensure 300+ DPI; Y-tip conversion scales with resolution |
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
    - Y-tip threshold too aggressive for image DPI
    - Cortex filtering removing legitimate structures

    **Solutions:**
    - Check image DPI is correctly detected (algorithm auto-adapts)
    - For low-DPI images (<150), Y-tip threshold is conservative (2px)
    - Use debug mode to verify DPI-aware parameter scaling
    - Manually edit before processing if needed

??? problem "Ripples not fully removed"
    **Causes:**
    - Inconsistent ripple pattern
    - Poor image quality
    - Ripples too thick
    - Y-tip artifacts creating junction remnants

    **Solutions:**
    - Improve scan quality (300+ DPI recommended)
    - Pre-process to enhance contrast
    - Algorithm automatically adjusts Y-tip removal (2-8px) based on DPI
    - Use debug mode to verify Y-tip junction conversion is working

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
   - Use 300+ DPI (optimal for algorithm performance)
   - Black and white mode
   - Clean scanner glass
   - **DPI Impact**: Higher DPI enables more precise Y-tip removal and cortex filtering

2. **Editing**
   - Remove text and scales
   - Fill gaps in lines
   - Increase contrast
   - Preserve DPI metadata for automatic algorithm adaptation

3. **Format**
   - Save as PNG with DPI metadata
   - Use lossless compression
   - Preserve metadata for DPI-aware processing


## Performance Optimization

### Memory Management
- Process images under 4000×4000 pixels
- Close unnecessary applications
- Use 64-bit Python

### Speed Improvements
- Resize large images first
- Process in batches overnight
- Use SSD for temp files
