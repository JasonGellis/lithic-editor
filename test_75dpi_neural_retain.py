#!/usr/bin/env python
"""Test neural upscaling on 75 DPI but retain 75 DPI output"""

import sys
import os
sys.path.insert(0, '.')

from lithic_editor.processing import ripple_removal

print("=" * 70)
print("NEURAL CLEANING TEST - 75 DPI Input, 75 DPI Output")
print("=" * 70)

print("\nTesting neural cleaning on 75 DPI image but retaining 75 DPI output")
print("Process: 75 DPI → Upscale to 300 DPI (neural) → Downscale back to 75 DPI")
print("Benefits: Neural cleaning + fast processing + small file size")
print("=" * 70)

# Test on a 75 DPI image
test_path = 'example_images/lithic_75dpi.png'  # Assuming you have this
output_folder = 'image_debug/lithic_75dpi_neural_retain'
os.makedirs(output_folder, exist_ok=True)

print('\nProcessing 75 DPI image WITH neural cleaning, retaining 75 DPI...')

# Use neural cleaning but output at 75 DPI
result = ripple_removal.process_lithic_drawing(
    test_path,
    output_folder=output_folder,
    dpi_info=75,  # Actual DPI
    default_dpi=75,
    save_debug=True,
    preserve_cortex=True,
    neural_cleaning=True,  # Enable neural cleaning
    neural_cleaning_dpi_range=(50, 400),  # Include 75 DPI
    neural_cleaning_target_dpi=75,  # Keep output at 75 DPI after neural cleaning
    upscale_model='espcn'
)

print('\nProcessing complete!')
print('Compare the results in:')
print('  - Standard 75 DPI: image_debug/lithic_75dpi/')
print('  - Neural cleaned:   image_debug/lithic_75dpi_neural_retain/')
print('\nThis should give you neural-cleaned quality at 75 DPI resolution')