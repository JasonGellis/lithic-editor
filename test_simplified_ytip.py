#!/usr/bin/env python
"""Test simplified Y-tip detection without dimension scaling"""

import sys
sys.path.insert(0, '.')

from lithic_editor.processing.ripple_removal import process_lithic_drawing

print('Testing simplified Y-tip threshold (no dimension scaling)...')
print('=' * 60)

# Test with 600 DPI image that will be downscaled to 300 DPI
result = process_lithic_drawing(
    'example_images/116.png',  # 600 DPI image
    output_folder='image_debug/116_simplified',
    save_debug=True,
    downscale_high_dpi=True,  # Enable downscaling
    high_dpi_threshold=300     # Threshold for downscaling (will trigger for 600 DPI)
)

print('=' * 60)
print('✓ Simplified Y-tip detection completed successfully!')
print(f'Output shape: {result.shape}')