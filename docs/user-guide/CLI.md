## Command Line Processing

### Basic Usage

```bash
# Process single image
lithic-editor process input.png

# Specify output directory
lithic-editor process input.png --output results/

# Enable debug mode
lithic-editor process input.png --debug
```

### Batch Processing

```bash
# Process all PNG files
for file in *.png; do
    lithic-editor process "$file" --output processed/
done

# Process with parallel jobs
find . -name "*.png" | parallel lithic-editor process {} --quiet
```

### Advanced Options

```bash
# Neural network upscaling with debug output
lithic-editor process low_res.png \
    --auto-upscale \
    --default-dpi 150 \
    --upscale-model fsrcnn \
    --upscale-threshold 300 \
    --debug

# Batch processing with upscaling
for file in *.png; do
    lithic-editor process "$file" \
        --auto-upscale \
        --default-dpi 200 \
        --output "processed/${file%.png}/"
done
```
