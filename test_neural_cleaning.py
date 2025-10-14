#!/usr/bin/env python
"""Test neural cleaning feature for 300 DPI images with artifacts"""

import sys
sys.path.insert(0, '.')

from lithic_editor.processing.ripple_removal import process_lithic_drawing
import os

print("=" * 70)
print("NEURAL CLEANING TEST - Remove Artifacts via Downscale-Upscale")
print("=" * 70)

print("\nThis test demonstrates neural cleaning on your 300 DPI lithic image")
print("that previously had 19 remaining ripples.")
print()
print("Process: 300 DPI → Downscale to 75 DPI → Upscale to 300 DPI (neural)")
print("Expected: Cleaner skeleton with fewer Y-tips, similar to 75 DPI upscaled")
print("=" * 70)

# Test on the problematic 300 DPI image
test_path = 'example_images/lithic_600dpi.png'
output_folder = 'image_debug/lithic_600dpiV2_neural_cleaned'
os.makedirs(output_folder, exist_ok=True)

print('\nProcessing 300 DPI lithic WITH neural cleaning...')
result = process_lithic_drawing(
    test_path,
    output_folder=output_folder,
    dpi_info=300,
    default_dpi=300,
    save_debug=True,
    preserve_cortex=True,
    neural_cleaning=True,  # Enable neural cleaning
    neural_cleaning_dpi_range=(200, 400),  # Apply to 300 DPI
    upscale_model='espcn'
)

print('\nProcessing complete!')
print('Compare the results in:')
print('  - Original 300 DPI: image_debug/lithic_300dpi/')
print('  - Neural cleaned:   image_debug/lithic_300dpi_neural_cleaned/')
print('\nThe neural cleaned version should have significantly fewer ripples.')