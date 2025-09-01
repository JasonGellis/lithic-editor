# CLI Reference

Complete command-line interface reference for Lithic Editor.

## Global Options

```bash
lithic-editor [OPTIONS] COMMAND [ARGS]
```

| Option | Short | Description |
|--------|-------|-------------|
| `--version` | | Show version and exit |
| `--help` | `-h` | Show help message |
| `--gui` | | Launch GUI directly |

## Commands

### gui

Launch the graphical user interface.

```bash
lithic-editor gui
```

**Examples:**
```bash
# Launch GUI
lithic-editor gui

# Alternative: use global flag
lithic-editor --gui
```

### process

Process lithic drawings from the command line.

```bash
lithic-editor process IMAGE_PATH [OPTIONS]
```

**Arguments:**
- `IMAGE_PATH` - Path to input lithic drawing image (required)

**Options:**

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--output` | `-o` | `image_debug` | Output directory |
| `--debug` | | `False` | Save debug images and processing steps |
| `--quiet` | `-q` | `False` | Suppress output |
| `--auto-upscale` | | `False` | Automatically upscale images below target DPI |
| `--default-dpi` | | `None` | Default DPI to assume for images without metadata |
| `--upscale-model` | | `espcn` | Model to use for upscaling (espcn, fsrcnn) |
| `--upscale-threshold` | | `300` | DPI threshold for upscaling |

**Examples:**

```bash
# Basic processing
lithic-editor process drawing.png

# Specify output directory
lithic-editor process drawing.png --output results/

# Save debug images and processing steps
lithic-editor process drawing.png --debug

# Quiet mode (no output)
lithic-editor process drawing.png --quiet

# Combine options
lithic-editor process artifact.png -o output/ --debug --quiet

# Neural network upscaling for low-DPI images
lithic-editor process low_dpi.png --auto-upscale --default-dpi 150

# Use FSRCNN model with custom threshold
lithic-editor process drawing.png --upscale-model fsrcnn --upscale-threshold 250
```

### docs

Access documentation.

```bash
lithic-editor docs [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--offline` | Serve documentation locally |

**Examples:**

```bash
# Open online documentation
lithic-editor docs

# Serve documentation locally
lithic-editor docs --offline
```

### help

Show detailed help information.

```bash
lithic-editor help [TOPIC]
```

**Topics:**
- `api` - Show Python API usage examples

**Examples:**

```bash
# Show general help
lithic-editor help

# Show API help
lithic-editor help api
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Command line syntax error |
| 3 | File not found |
| 4 | Permission denied |

## Environment Variables

### LITHIC_OUTPUT_DIR
Default output directory for processed images.

```bash
export LITHIC_OUTPUT_DIR=/path/to/output
lithic-editor process image.png  # Uses LITHIC_OUTPUT_DIR
```

### LITHIC_DEBUG
Enable debug mode by default.

```bash
export LITHIC_DEBUG=1
lithic-editor process image.png  # Debug enabled
```

## Shell Completion

### Bash

Add to `~/.bashrc`:

```bash
eval "$(_LITHIC_EDITOR_COMPLETE=bash_source lithic-editor)"
```

### Zsh

Add to `~/.zshrc`:

```bash
eval "$(_LITHIC_EDITOR_COMPLETE=zsh_source lithic-editor)"
```

### Fish

Add to `~/.config/fish/completions/lithic-editor.fish`:

```bash
eval (env _LITHIC_EDITOR_COMPLETE=fish_source lithic-editor)
```

## Batch Processing

### Using Shell Loops

```bash
# Process all PNG files
for file in *.png; do
    lithic-editor process "$file" --output "processed/${file%.png}/"
done

# Process with parallel
find . -name "*.png" | parallel -j 4 lithic-editor process {} --output {.}/

# Process and log results
for file in drawings/*.png; do
    echo "Processing $file..."
    if lithic-editor process "$file" --quiet; then
        echo "✓ $file" >> success.log
    else
        echo "✗ $file" >> failed.log
    fi
done
```

### Using Find and Xargs

```bash
# Process all images recursively
find . -type f \( -name "*.png" -o -name "*.jpg" \) \
    -exec lithic-editor process {} --output {}_processed/ \;

# Parallel processing with xargs
find drawings/ -name "*.png" -print0 | \
    xargs -0 -n 1 -P 4 -I {} lithic-editor process {} --quiet
```

## Scripting Examples

### Processing Script

```bash
#!/bin/bash
# process_lithics.sh - Batch process lithic drawings

INPUT_DIR="${1:-./drawings}"
OUTPUT_DIR="${2:-./processed}"
LOG_FILE="processing.log"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Initialize log
echo "Processing started: $(date)" > "$LOG_FILE"

# Process counter
SUCCESS=0
FAILED=0

# Process each image
for image in "$INPUT_DIR"/*.{png,jpg,jpeg,tif,tiff} 2>/dev/null; do
    [ -f "$image" ] || continue
    
    basename=$(basename "$image")
    echo "Processing: $basename"
    
    if lithic-editor process "$image" \
        --output "$OUTPUT_DIR/${basename%.*}" \
        --debug --quiet; then
        ((SUCCESS++))
        echo "✓ $basename" >> "$LOG_FILE"
    else
        ((FAILED++))
        echo "✗ $basename" >> "$LOG_FILE"
    fi
done

# Summary
echo "Completed: $SUCCESS successful, $FAILED failed" | tee -a "$LOG_FILE"
```

### Watch Folder Script

```bash
#!/bin/bash
# watch_folder.sh - Auto-process new images

WATCH_DIR="${1:-./incoming}"
OUTPUT_DIR="${2:-./processed}"

echo "Watching $WATCH_DIR for new images..."

# Using inotify (Linux)
inotifywait -m -e create -e moved_to "$WATCH_DIR" |
while read -r directory event filename; do
    if [[ "$filename" =~ \.(png|jpg|jpeg|tif|tiff)$ ]]; then
        echo "Processing new file: $filename"
        lithic-editor process "$WATCH_DIR/$filename" \
            --output "$OUTPUT_DIR/${filename%.*}"
    fi
done

# Using fswatch (macOS)
fswatch -0 "$WATCH_DIR" | while read -d "" path; do
    filename=$(basename "$path")
    if [[ "$filename" =~ \.(png|jpg|jpeg|tif|tiff)$ ]]; then
        echo "Processing: $filename"
        lithic-editor process "$path" \
            --output "$OUTPUT_DIR/${filename%.*}"
    fi
done
```

## Docker Usage

```dockerfile
# Dockerfile
FROM python:3.9
RUN pip install git+https://github.com/JasonGellis/lithic-editor.git
ENTRYPOINT ["lithic-editor"]
```

```bash
# Build image
docker build -t lithic-editor .

# Process image
docker run -v $(pwd):/data lithic-editor \
    process /data/drawing.png --output /data/output/

# Run GUI (requires X11)
docker run -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $(pwd):/data \
    lithic-editor gui
```

## Performance Optimization

### Memory Management

```bash
# Limit memory usage
ulimit -v 2097152  # 2GB limit
lithic-editor process large_image.png

# Nice level for background processing
nice -n 19 lithic-editor process image.png
```

### Parallel Processing

```bash
# GNU Parallel
parallel -j 4 lithic-editor process {} ::: *.png

# Custom parallel script
#!/bin/bash
MAX_JOBS=4
for file in *.png; do
    while [ $(jobs -r | wc -l) -ge $MAX_JOBS ]; do
        sleep 1
    done
    lithic-editor process "$file" &
done
wait
```

## Troubleshooting

### Debug Mode

```bash
# Enable verbose output
lithic-editor process image.png --debug

# Check version and environment
lithic-editor --version

# Test with sample image
lithic-editor process --help
```

### Common Issues

**Permission Denied:**
```bash
# Check permissions
ls -la image.png
# Fix permissions
chmod 644 image.png
```

**Output Directory Issues:**
```bash
# Create output directory first
mkdir -p output/
lithic-editor process image.png --output output/
```

**Large File Processing:**
```bash
# Increase timeout for large files
timeout 300 lithic-editor process large_image.tiff
```

## Integration Examples

### Makefile

```makefile
# Makefile for lithic processing

INPUT_DIR = drawings
OUTPUT_DIR = processed
IMAGES = $(wildcard $(INPUT_DIR)/*.png)
OUTPUTS = $(patsubst $(INPUT_DIR)/%.png,$(OUTPUT_DIR)/%/9_high_quality.png,$(IMAGES))

all: $(OUTPUTS)

$(OUTPUT_DIR)/%/9_high_quality.png: $(INPUT_DIR)/%.png
	@mkdir -p $(dir $@)
	lithic-editor process $< --output $(OUTPUT_DIR)/$*

clean:
	rm -rf $(OUTPUT_DIR)

.PHONY: all clean
```

### Git Hook

```bash
#!/bin/bash
# .git/hooks/pre-commit
# Process lithic images before commit

for file in $(git diff --cached --name-only | grep -E '\.(png|jpg)$'); do
    if [[ "$file" == drawings/* ]]; then
        echo "Processing $file..."
        lithic-editor process "$file" --output "processed/${file#drawings/}"
        git add "processed/${file#drawings/}"
    fi
done
```