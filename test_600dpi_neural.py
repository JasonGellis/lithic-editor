#!/usr/bin/env python
"""Test neural cleaning on 600 DPI image"""

import sys
import os
sys.path.insert(0, '.')

# Import just what we need
from lithic_editor.processing import ripple_removal

print("=" * 70)
print("NEURAL CLEANING TEST - 600 DPI Image")
print("=" * 70)

print("\nTesting neural cleaning on 600 DPI image (368.png)")
print("Process: 600 DPI → Downscale to 75 DPI → Upscale back to 600 DPI (neural)")
print("Note: Image will be returned to original 600 DPI resolution after cleaning")
print("=" * 70)

# Test on the 600 DPI image
test_path = 'example_images/369.png'
output_folder = 'image_debug/369_neural_cleaned'
os.makedirs(output_folder, exist_ok=True)

print('\nProcessing 600 DPI image WITH neural cleaning...')
result = ripple_removal.process_lithic_drawing(
    test_path,
    output_folder=output_folder,
    dpi_info=600,  # Correct DPI
    default_dpi=600,
    save_debug=True,
    preserve_cortex=True,
    neural_cleaning=True,  # Enable neural cleaning
    neural_cleaning_dpi_range=(200, 650),  # Expanded range to include 600 DPI
    upscale_model='espcn'
)

print('\nProcessing complete!')
print('Compare the results in:')
print('  - Native 600 DPI:   image_debug/368/')
print('  - Neural cleaned:   image_debug/368_neural_cleaned/')
print('\nNote: Neural cleaning now preserves original 600 DPI resolution')