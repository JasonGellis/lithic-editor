import cv2
import numpy as np
import networkx as nxgraph
from skimage import morphology
from scipy.ndimage import label
import os
import traceback
from PIL import Image
import numpy as np
from .upscaling import (
    detect_image_dpi, needs_upscaling, upscale_image_to_target_dpi,
    validate_upscaling_inputs
)


def exception_hook(exc_type, exc_value, exc_traceback):
    print("Exception hook triggered:")
    print("Type:", exc_type)
    print("Value:", exc_value)
    print("Traceback:", "".join(traceback.format_tb(exc_traceback)))
    # Write to a file for inspection
    with open("error_log.txt", "w") as f:
        f.write("Exception hook triggered:\n")
        f.write(f"Type: {exc_type}\n")
        f.write(f"Value: {exc_value}\n")
        f.write(f"Traceback: {''.join(traceback.format_tb(exc_traceback))}")
    sys.__excepthook__(exc_type, exc_value, exc_traceback)

# Install the exception hook
import sys
sys.excepthook = exception_hook

print("Starting Lithic Editor GUI")


def improve_line_quality_antialias(binary_image, line_boost=1.0, preserve_thickness=True):
    """
    Improve line quality using advanced anti-aliasing while preserving original thickness

    Args:
        binary_image: Input binary image (black lines on white background)
        line_boost: Factor to slightly adjust line presence (1.0 = no change)
        preserve_thickness: Whether to preserve original line thickness

    Returns:
        improved_image: Enhanced image with better line quality
    """
    # Ensure correct format
    if binary_image.max() <= 1:
        binary_image = binary_image * 255

    # Convert to proper format
    binary_image = binary_image.astype(np.uint8)

    if preserve_thickness:
        # Gentle enhancement that preserves thickness
        scale_factor = 3  # Reduced from 6 for better thickness preservation
        h, w = binary_image.shape
        upscaled = cv2.resize(binary_image, (w * scale_factor, h * scale_factor),
                             interpolation=cv2.INTER_CUBIC)

        # Very light morphological operations
        if line_boost > 1.0:
            kernel = np.ones((2, 2), np.uint8)
            upscaled = cv2.dilate(upscaled, kernel, iterations=1)

        # Gentle Gaussian blur for smoothing
        blurred = cv2.GaussianBlur(upscaled, (3, 3), 0.5)  # Reduced sigma

        # Conservative threshold to maintain line width
        _, smoothed = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)  # Lower threshold

        # Light median filter
        smoothed = cv2.medianBlur(smoothed, 3)

        # Scale back down with high-quality interpolation
        result = cv2.resize(smoothed, (w, h), interpolation=cv2.INTER_AREA)

        # Final threshold with preservation bias
        _, result = cv2.threshold(result, 180, 255, cv2.THRESH_BINARY)  # Lower threshold

    else:
        # Original aggressive enhancement for very thin lines
        scale_factor = 6
        h, w = binary_image.shape
        upscaled = cv2.resize(binary_image, (w * scale_factor, h * scale_factor),
                             interpolation=cv2.INTER_CUBIC)

        kernel = np.ones((2, 2), np.uint8)
        if line_boost > 1.0:
            upscaled = cv2.dilate(upscaled, kernel, iterations=1)

        blurred = cv2.GaussianBlur(upscaled, (3, 3), 0.8)
        _, smoothed = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)
        smoothed = cv2.medianBlur(smoothed, 3)

        if line_boost > 1.3:
            smoothed = cv2.erode(smoothed, kernel, iterations=1)

        result = cv2.resize(smoothed, (w, h), interpolation=cv2.INTER_AREA)
        _, result = cv2.threshold(result, 200, 255, cv2.THRESH_BINARY)

    return result

def create_thickness_aware_mask(original_binary, structural_mask, min_thickness=1, max_thickness=10):
    """
    Create a thickness-aware reconstruction mask that preserves original line widths

    Args:
        original_binary: Original binary image (boolean array, True=foreground)
        structural_mask: Skeleton-based mask of structural elements (boolean array)
        min_thickness: Minimum line thickness to preserve
        max_thickness: Maximum line thickness to preserve

    Returns:
        final_mask: Boolean mask preserving original line thickness for structural elements
    """
    # Convert structural mask to distance transform to get reconstruction zones
    from scipy.ndimage import distance_transform_edt

    # Create distance transform from structural skeleton
    # This gives us zones around structural lines where we should preserve original content
    structural_distance = distance_transform_edt(~structural_mask)

    # Create adaptive dilation zones based on local line thickness in original
    original_distance = distance_transform_edt(~original_binary)

    # For each structural pixel, determine appropriate reconstruction radius
    reconstruction_mask = np.zeros_like(original_binary, dtype=bool)

    # Get coordinates of structural skeleton pixels
    struct_y, struct_x = np.where(structural_mask)

    for i in range(len(struct_y)):
        y, x = struct_y[i], struct_x[i]

        # Find original line thickness at this location by looking in neighborhood
        # Get local neighborhood around this skeleton point
        neighborhood_size = max_thickness + 2
        y_min = max(0, y - neighborhood_size)
        y_max = min(original_binary.shape[0], y + neighborhood_size + 1)
        x_min = max(0, x - neighborhood_size)
        x_max = min(original_binary.shape[1], x + neighborhood_size + 1)

        # Get the original content in this neighborhood
        local_original = original_binary[y_min:y_max, x_min:x_max]

        if np.any(local_original):
            # Find the maximum distance to background in this neighborhood
            # This approximates the local line thickness
            local_distances = distance_transform_edt(local_original)
            local_thickness = np.max(local_distances)

            # Clamp thickness to reasonable bounds
            reconstruction_radius = max(min_thickness, min(max_thickness, int(local_thickness * 0.2)))

            # Create circular reconstruction zone around this skeleton point
            for dy in range(-reconstruction_radius, reconstruction_radius + 1):
                for dx in range(-reconstruction_radius, reconstruction_radius + 1):
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < original_binary.shape[0] and
                        0 <= nx < original_binary.shape[1] and
                        dy*dy + dx*dx <= reconstruction_radius*reconstruction_radius):
                        reconstruction_mask[ny, nx] = True

    # Final mask: original content AND within reconstruction zones
    final_mask = original_binary & reconstruction_mask

    return final_mask

def crop_to_content(image, padding=10):
    """Crop image to content plus padding"""
    # If image is grayscale, convert to 3 channels for consistent handling
    if len(image.shape) == 2:
        img_for_crop = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        img_for_crop = image.copy()

    # Convert to grayscale for finding bounds
    gray = cv2.cvtColor(img_for_crop, cv2.COLOR_BGR2GRAY)

    # Threshold to binary
    _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get bounding box for all contours
    if contours:
        x_min, y_min = image.shape[1], image.shape[0]
        x_max, y_max = 0, 0

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)

        # Add padding
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(image.shape[1], x_max + padding)
        y_max = min(image.shape[0], y_max + padding)

        # Crop the image
        return image[y_min:y_max, x_min:x_max]

    # If no contours found, return original
    return image


def debug_image_info(name, img):
    """Print detailed image information for debugging"""
    if img is None:
        print(f"{name:<30} {'N/A':>10} {'N/A':>10} {'None':>15} {'N/A':>8} {'N/A':>8}")
        return

    if isinstance(img, bool) or (hasattr(img, 'dtype') and img.dtype == bool):
        img_type = "Boolean"
    elif not hasattr(img, 'dtype'):
        img_type = type(img).__name__
    else:
        img_type = str(img.dtype)

    shape_info = "N/A"
    h, w = "N/A", "N/A"
    min_val, max_val = "N/A", "N/A"

    if hasattr(img, 'shape'):
        if len(img.shape) == 2:
            h, w = img.shape
            shape_info = f"2D: {w}x{h}"
        elif len(img.shape) == 3:
            h, w, c = img.shape
            shape_info = f"3D: {w}x{h}x{c}"
        else:
            shape_info = f"Other: {img.shape}"

    if hasattr(img, 'min') and hasattr(img, 'max'):
        try:
            min_val = f"{img.min():.2f}"
            max_val = f"{img.max():.2f}"
        except:
            min_val = "Error"
            max_val = "Error"

    print(f"{name:<30} {w:>10} {h:>10} {img_type:>15} {min_val:>8} {max_val:>8}")
    return img

    # Save the image
    cv2.imwrite(output_path, img)
    print(f"Debug image saved to {output_path}")

def separate_cortex_and_structure(binary_image, preserve_cortex=True, cortex_size_threshold=20):
    """
    Separate cortex stippling from structural lines before skeletonization.
    
    Args:
        binary_image: Binary image (0=background, 255=foreground)
        preserve_cortex: If True, separates cortex; if False, processes everything together
        cortex_size_threshold: Maximum pixel area for cortex components
        
    Returns:
        Tuple of (structural_image, cortex_mask)
    """
    if not preserve_cortex:
        # Process everything together - no cortex separation
        return binary_image, np.zeros_like(binary_image, dtype=bool)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    
    # Create separate images for cortex and structural elements
    structural_image = np.zeros_like(binary_image)
    cortex_mask = np.zeros_like(binary_image, dtype=bool)
    
    cortex_count = 0
    structural_count = 0
    
    # Process each component (skip background label 0)
    for label in range(1, num_labels):
        component_area = stats[label, cv2.CC_STAT_AREA]
        component_mask = labels == label
        
        if component_area <= cortex_size_threshold:
            # Small component - classify as cortex stippling
            cortex_mask[component_mask] = True
            cortex_count += 1
        else:
            # Large component - structural line
            structural_image[component_mask] = 255
            structural_count += 1
    
    print(f"Separated: {structural_count} structural components, {cortex_count} cortex stipples")
    return structural_image, cortex_mask


def process_lithic_drawing(image_path, output_folder="image_debug", dpi_info=None, format_info=None, output_dpi=None, save_debug=False,
                          upscale_low_dpi=False, default_dpi=None, upscale_model='espcn', target_dpi=300,
                          scale_image_path=None, return_scale_factor=False, debug_filename=None, preserve_cortex=True):
    """
    Process a lithic drawing to remove ripple lines while preserving original line quality and metadata

    Args:
        image_path: Path to the input image
        output_folder: Folder to save all output images
        dpi_info: DPI information to preserve (tuple of x,y dpi)
        format_info: Original image format to preserve
        output_dpi: DPI for output images
        save_debug: Whether to save debug images
        upscale_low_dpi: Whether to upscale images below target_dpi
        default_dpi: DPI to assume if metadata missing (for upscaling decisions)
        upscale_model: Model to use for upscaling ('espcn' or 'fsrcnn')
        target_dpi: Target DPI for upscaling (default: 300)
        scale_image_path: Optional path to scale image (processed with same factor)
        return_scale_factor: Whether to return upscaling factor in results
        preserve_cortex: Whether to preserve cortex stippling (default: True)

    Returns:
        cleaned_image: Image with ripple lines removed but original line quality preserved
        If return_scale_factor=True, returns dict with image and scale factor
    """
    # Only create output folder if saving debug images
    if save_debug and output_folder:
        os.makedirs(output_folder, exist_ok=True)
        print(f"Debug mode enabled. Output folder: {output_folder}")

    # Extract base filename for debug images
    if debug_filename:
        base_filename = debug_filename
    elif isinstance(image_path, str):
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
    else:
        base_filename = "image"

    # Add debugging header
    print("\n=== IMAGE DIMENSIONS DEBUGGING ===")
    print(f"{'Step':<30} {'Width':>10} {'Height':>10} {'Type':>15} {'Min':>8} {'Max':>8}")
    print("-" * 80)

    # Step 1: Read the image and preserve metadata
    print("Reading image...")
    if isinstance(image_path, str):
        # Use Pillow to read the image and preserve metadata
        try:
            pil_image = Image.open(image_path)
            # Store metadata if not already provided
            if format_info is None:
                format_info = pil_image.format
                print(f"Image format: {format_info}")

            if dpi_info is None and hasattr(pil_image, 'info') and 'dpi' in pil_image.info:
                dpi_info = pil_image.info['dpi']
                print(f"Image DPI: {dpi_info}")

            # Convert to numpy array for processing (grayscale)
            original_image = np.array(pil_image.convert('L'))
            print(f"Image read successfully. Shape: {original_image.shape}")
        except Exception as e:
            print(f"Error reading with Pillow: {e}")
            # Fallback to OpenCV
            original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if original_image is None:
                raise ValueError(f"Could not read image at {image_path}")
            print(f"Image read with OpenCV. Shape: {original_image.shape}")
    else:
        # Assume image_path is already a numpy array
        original_image = image_path
        print(f"Using provided numpy array. Shape: {original_image.shape}")

    # Handle upscaling if requested
    scale_factor = 1.0
    processed_scale = None

    if upscale_low_dpi:
        # Determine current DPI
        current_dpi = None
        if dpi_info:
            if isinstance(dpi_info, tuple):
                current_dpi = max(dpi_info[0], dpi_info[1])
            else:
                current_dpi = int(dpi_info)
        elif default_dpi:
            current_dpi = default_dpi
            print(f"Using default DPI: {current_dpi} (no metadata found)")
        else:
            # For non-interactive mode, we can't proceed without DPI
            print("Warning: No DPI information found and no default_dpi provided. Skipping upscaling.")
            upscale_low_dpi = False

        if upscale_low_dpi and current_dpi:
            # Validate upscaling parameters
            is_valid, error_msg = validate_upscaling_inputs(current_dpi, target_dpi, upscale_model)
            if not is_valid:
                print(f"Upscaling validation failed: {error_msg}")
                upscale_low_dpi = False
            elif needs_upscaling(current_dpi, target_dpi):
                print(f"Image DPI ({current_dpi}) below target ({target_dpi}). Upscaling...")

                # Save original low-DPI image for comparison
                if save_debug:
                    os.makedirs(output_folder, exist_ok=True)  # Ensure folder exists before upscaling
                    save_debug_image(original_image, os.path.join(output_folder, f'0_{base_filename}_original_low_dpi.png'),
                                    f'Input ({current_dpi} DPI)', dpi_info, format_info, (current_dpi, current_dpi) if isinstance(current_dpi, int) else current_dpi)

                # Upscale the main image
                upscaled_image, scale_factor = upscale_image_to_target_dpi(
                    original_image, current_dpi, target_dpi, upscale_model
                )
                original_image = upscaled_image

                # Update DPI info to reflect upscaling
                new_dpi = int(current_dpi * scale_factor)
                if isinstance(dpi_info, tuple):
                    dpi_info = (new_dpi, new_dpi)
                else:
                    dpi_info = new_dpi

                print(f"Upscaling completed: {current_dpi} DPI → {new_dpi} DPI (factor: {scale_factor:.1f}x)")

                # Process scale image if provided
                if scale_image_path:
                    print(f"Processing scale image with same factor...")
                    try:
                        scale_pil = Image.open(scale_image_path)
                        scale_array = np.array(scale_pil.convert('L'))
                        processed_scale, _ = upscale_image_to_target_dpi(
                            scale_array, current_dpi, target_dpi, upscale_model
                        )
                        if save_debug:
                            save_debug_image(processed_scale, os.path.join(output_folder, f'0b_{base_filename}_upscaled_scale.png'),
                                            f'Upscaled Scale ({new_dpi} DPI)', (new_dpi, new_dpi), 'PNG', (new_dpi, new_dpi))
                    except Exception as e:
                        print(f"Error processing scale image: {e}")

                # Save upscaled main image
                if save_debug:
                    save_debug_image(original_image, os.path.join(output_folder, f'0a_{base_filename}_upscaled_300dpi.png'),
                                    f'Upscaled ({new_dpi} DPI)', dpi_info, format_info, (new_dpi, new_dpi) if isinstance(new_dpi, int) else new_dpi)
            else:
                print(f"Image DPI ({current_dpi}) already meets target ({target_dpi}). No upscaling needed.")

    # Print metadata info
    if dpi_info:
        print(f"Original DPI information: {dpi_info}")
    if output_dpi:
        print(f"Output DPI will be set to: {output_dpi}")
    elif dpi_info:
        print(f"Output DPI will match original: {dpi_info}")
    else:
        print("Output DPI will be unset (application defaults will apply)")
    if format_info:
        print(f"Preserving original format: {format_info}")

    # Save the original image
    if save_debug:
        save_debug_image(original_image, os.path.join(output_folder, f'1_{base_filename}_original_image.png'),
                        'Original Image', dpi_info, format_info, output_dpi)

    # Step 2: Preprocess the image
    print("Preprocessing image...")

    # Threshold to binary (if necessary)
    if original_image.max() > 1:  # Check if image is not already binary
        _, binary = cv2.threshold(original_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        binary = binary > 0  # Convert to boolean
        print("Image thresholded to binary")
    else:
        binary = original_image > 0
        print("Image already binary")

    # Save a copy of the binary image for later reconstruction
    binary_image = binary.astype(np.uint8) * 255

    # Separate cortex stippling from structural lines
    structural_image, cortex_mask = separate_cortex_and_structure(binary_image, preserve_cortex=preserve_cortex)
    
    # Save debug images for cortex separation
    if save_debug and preserve_cortex:
        # Show separated structural elements
        save_debug_image(structural_image, os.path.join(output_folder, f'1a_{base_filename}_structural_only.png'),
                        'Structural Lines Only', dpi_info, format_info, output_dpi)
        
        # Show cortex mask
        cortex_debug = cortex_mask.astype(np.uint8) * 255
        save_debug_image(cortex_debug, os.path.join(output_folder, f'1b_{base_filename}_cortex_mask.png'),
                        'Cortex Stippling Mask', dpi_info, format_info, output_dpi)

    # Skeletonize only the structural elements (cortex bypasses skeletonization)
    print("Skeletonizing structural elements...")
    if preserve_cortex:
        # Skeletonize only structural lines
        structural_binary = structural_image > 0
        skeleton = morphology.skeletonize(structural_binary)
        # Update binary_image to reflect structural-only processing
        binary_image = structural_image
        binary = structural_binary
    else:
        # Process everything together (original behavior)
        binary = binary_image > 0
        skeleton = morphology.skeletonize(binary)
    skeleton_img = skeleton.astype(np.uint8) * 255
    print(f"Skeleton created. Non-zero pixels: {np.count_nonzero(skeleton)}")

    # Save the skeleton image
    if save_debug:
        save_debug_image(skeleton_img, os.path.join(output_folder, f'2_{base_filename}_skeleton.png'),
                        'Skeleton', dpi_info, format_info, output_dpi)

    # Step 3: Find endpoints and junctions
    print("Finding endpoints and junctions...")
    height, width = skeleton.shape

    # Initialize arrays to store special points
    endpoints = []
    junctions = []

    # Get coordinates of skeleton pixels
    y_coords, x_coords = np.where(skeleton)
    print(f"Total skeleton pixels: {len(y_coords)}")

    # For each skeleton pixel, count neighbors to determine if it's an endpoint or junction
    for i in range(len(y_coords)):
        y, x = y_coords[i], x_coords[i]

        # Skip border pixels
        if y > 0 and y < height-1 and x > 0 and x < width-1:
            # Get 8-connected neighborhood
            neighborhood = skeleton[y-1:y+2, x-1:x+2].flatten()
            # Remove the center pixel (which is the pixel itself)
            neighborhood = np.delete(neighborhood, 4)
            # Count non-zero neighbors
            num_neighbors = np.count_nonzero(neighborhood)

            if num_neighbors == 1:
                # This is an endpoint
                endpoints.append((x, y))
            elif num_neighbors >= 3:
                # This is a junction
                junctions.append((x, y))

    print(f"Found {len(endpoints)} endpoints and {len(junctions)} junctions")

    # Create visualization of endpoints and junctions
    debug_img = np.zeros((height, width, 3), dtype=np.uint8)
    debug_img[skeleton] = [255, 255, 255]  # White for skeleton

    # Use small markers
    for x, y in endpoints:
        cv2.circle(debug_img, (x, y), 1, (0, 0, 255), -1)  # Red for endpoints

    for x, y in junctions:
        cv2.circle(debug_img, (x, y), 1, (0, 255, 0), -1)  # Green for junctions

    if save_debug:
        save_debug_image(debug_img, os.path.join(output_folder, f'3_{base_filename}_endpoints_junctions.png'),
                        'Endpoints & Junctions', dpi_info, format_info, output_dpi)


    # Step 4: Identify line segments by removing junctions and endpoints
    print("Identifying line segments...")

    # Create a mask for special points (endpoints and junctions)
    special_points_mask = np.zeros_like(skeleton, dtype=bool)
    for x, y in endpoints + junctions:
        special_points_mask[y, x] = True

    # Remove special points from skeleton to get segments
    segments = skeleton.copy()
    segments[special_points_mask] = False

    # Label the connected components (segments)
    structure = np.ones((3, 3), dtype=np.int32)  # 8-connectivity
    labeled_segments, num_segments = label(segments, structure=structure)
    print(f"Found {num_segments} line segments")

    # Create a colored visualization of the labeled segments
    segment_colors = np.zeros((height, width, 3), dtype=np.uint8)

    # Generate random colors for each segment
    np.random.seed(42)  # For reproducibility
    colors = np.random.randint(50, 256, size=(num_segments + 1, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # Background is black

    # Apply colors to segments
    for y in range(height):
        for x in range(width):
            segment_id = labeled_segments[y, x]
            segment_colors[y, x] = colors[segment_id]

    if save_debug:
        save_debug_image(segment_colors, os.path.join(output_folder, f'4_{base_filename}_labeled_segments.png'),
                        f'Labeled Segments ({num_segments})', dpi_info, format_info, output_dpi)

    # Step 5: Build graph representation
    print("Building graph representation...")
    G = nxgraph.Graph()
    # Add nodes for endpoints and junctions
    for i, (x, y) in enumerate(endpoints):
        G.add_node(f"e{i}", type="endpoint", pos=(x, y))

    for i, (x, y) in enumerate(junctions):
        G.add_node(f"j{i}", type="junction", pos=(x, y))

    # Create a mapping from coordinates to node IDs
    coord_to_node = {}
    for node, data in G.nodes(data=True):
        coord_to_node[data['pos']] = node

    # For each segment, find which special points it connects
    segment_connections = {}
    for segment_id in range(1, num_segments + 1):
        # Get mask for this segment
        segment_mask = labeled_segments == segment_id

        # Dilate to find connecting special points
        dilated_mask = morphology.binary_dilation(segment_mask, np.ones((3, 3), dtype=np.uint8))

        # Find all special points connected to this segment
        connected_points = []

        # Check endpoints
        for point in endpoints:
            x, y = point
            if dilated_mask[y, x]:
                connected_points.append(coord_to_node[(x, y)])

        # Check junctions
        for point in junctions:
            x, y = point
            if dilated_mask[y, x]:
                connected_points.append(coord_to_node[(x, y)])

        # Store the connections for this segment
        if len(connected_points) >= 2:
            segment_connections[segment_id] = connected_points

            # Add edges between all pairs of connected points
            for i in range(len(connected_points)):
                for j in range(i+1, len(connected_points)):
                    G.add_edge(connected_points[i], connected_points[j], segment=segment_id)

    # Step 6: Identify ripple lines
    print("Identifying ripple lines...")

    # A segment is a ripple line if it connects to at least one endpoint
    ripple_segments = set()

    for segment_id, connected_points in segment_connections.items():
        # Check if any connected point is an endpoint
        has_endpoint = any(node.startswith('e') for node in connected_points)

        if has_endpoint:
            ripple_segments.add(segment_id)

    print(f"Identified {len(ripple_segments)} ripple segments")

    # Create visualization of ripple segments
    ripple_viz = np.zeros((height, width, 3), dtype=np.uint8)

    # Draw structural segments in white
    for segment_id in range(1, num_segments + 1):
        if segment_id not in ripple_segments:
            segment_mask = labeled_segments == segment_id
            ripple_viz[segment_mask] = [255, 255, 255]

    # Draw ripple segments in red
    for segment_id in ripple_segments:
        segment_mask = labeled_segments == segment_id
        ripple_viz[segment_mask] = [0, 0, 255]

    # Mark endpoints and junctions
    for x, y in endpoints:
        cv2.circle(ripple_viz, (x, y), 1, (0, 0, 255), -1)  # Red for endpoints

    for x, y in junctions:
        cv2.circle(ripple_viz, (x, y), 1, (0, 255, 0), -1)  # Green for junctions

    if save_debug:
        save_debug_image(ripple_viz, os.path.join(output_folder, f'5_{base_filename}_ripple_identification.png'),
                        'Ripple vs Structural', dpi_info, format_info, output_dpi)

    # Step 7: Create mask of structural elements (excluding ripple endpoints)
    print("Creating structural mask...")

    # Start with clean binary mask
    structural_mask = np.zeros_like(skeleton, dtype=bool)

    # Add non-ripple segments to the mask
    for segment_id in range(1, num_segments + 1):
        if segment_id not in ripple_segments:
            segment_mask = labeled_segments == segment_id
            structural_mask = structural_mask | segment_mask

    # Add junction points (single pixel) to ensure connectivity
    for x, y in junctions:
        structural_mask[y, x] = True

    # Filter out endpoints that connect to ripple segments to preserve line quality
    # Only add endpoints that connect to structural segments
    structural_endpoints = []
    for x, y in endpoints:
        # Check if this endpoint connects to any structural segments
        # by looking in a small neighborhood
        connects_to_structural = False
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if (0 <= ny < height and 0 <= nx < width and
                    labeled_segments[ny, nx] > 0 and
                    labeled_segments[ny, nx] not in ripple_segments):
                    connects_to_structural = True
                    break
            if connects_to_structural:
                break

        # Only add endpoint if it connects to structural elements
        if connects_to_structural:
            structural_mask[y, x] = True
            structural_endpoints.append((x, y))

    print(f"Filtered endpoints: {len(endpoints)} total -> {len(structural_endpoints)} structural")

    # Create a skeleton-based cleaned image for comparison
    skeleton_cleaned = np.zeros_like(skeleton_img)
    skeleton_cleaned[structural_mask] = 255

    # INVERT the skeleton-based cleaned image (black lines on white background)
    skeleton_cleaned_inverted = 255 - skeleton_cleaned

    # Save the skeleton-based cleaned image (inverted)
    if save_debug:
        save_debug_image(skeleton_cleaned_inverted, os.path.join(output_folder, f'6_{base_filename}_skeleton_cleaned.png'),
                        'Skeleton Cleaned', dpi_info, format_info, output_dpi)

    # NEW: Create debug image showing filtered endpoints
    filtered_endpoints_viz = np.zeros((height, width, 3), dtype=np.uint8)
    filtered_endpoints_viz[skeleton] = [100, 100, 100]  # Gray for all skeleton

    # Mark structural segments in white
    for segment_id in range(1, num_segments + 1):
        if segment_id not in ripple_segments:
            segment_mask = labeled_segments == segment_id
            filtered_endpoints_viz[segment_mask] = [255, 255, 255]

    # Mark junctions in green
    for x, y in junctions:
        cv2.circle(filtered_endpoints_viz, (x, y), 2, (0, 255, 0), -1)

    # Mark structural endpoints in blue
    for x, y in structural_endpoints:
        cv2.circle(filtered_endpoints_viz, (x, y), 2, (255, 0, 0), -1)

    # Mark filtered-out endpoints in red
    for x, y in endpoints:
        if (x, y) not in structural_endpoints:
            cv2.circle(filtered_endpoints_viz, (x, y), 2, (0, 0, 255), -1)

    if save_debug:
        if save_debug:
            save_debug_image(filtered_endpoints_viz, os.path.join(output_folder, f'6a_{base_filename}_endpoint_filtering.png'),
                            'Endpoint Filtering', dpi_info, format_info, output_dpi)

    # Step 8: Create final image with hybrid thickness preservation
    print("Creating final image with hybrid thickness preservation...")

    # Convert binary_image to boolean for processing
    binary_bool = binary_image > 0

    # Use the new thickness-aware reconstruction
    thickness_preserved_mask = create_thickness_aware_mask(
        original_binary=binary_bool,
        structural_mask=structural_mask,
        min_thickness=1,
        max_thickness=2
    )

    # Create the final cleaned image
    final_cleaned = np.zeros_like(binary_image)
    final_cleaned[thickness_preserved_mask] = 255

    # Apply slight morphological closing to connect any small gaps
    kernel_close = np.ones((2, 2), np.uint8)
    final_cleaned = cv2.morphologyEx(final_cleaned, cv2.MORPH_CLOSE, kernel_close)

    # Add cortex stippling back to final result if preserving
    if preserve_cortex and 'cortex_mask' in locals():
        print("Adding cortex stippling back to final result...")
        final_cleaned[cortex_mask] = 255
        print(f"Cortex pixels added: {np.count_nonzero(cortex_mask)}")

    # INVERT the final cleaned image (black lines on white background)
    final_cleaned_inverted = 255 - final_cleaned

    # Save the final cleaned image (inverted)
    if save_debug:
        save_debug_image(final_cleaned_inverted, os.path.join(output_folder, f'7_{base_filename}_final_cleaned.png'),
                        'Final Cleaned', dpi_info, format_info, output_dpi)

    print("Processing complete!")

    # Return results based on requested format
    if return_scale_factor or scale_image_path:
        result = {
            'processed_image': final_cleaned_inverted,
            'scale_factor': scale_factor,
            'original_dpi': current_dpi if 'current_dpi' in locals() else None,
            'final_dpi': int(current_dpi * scale_factor) if 'current_dpi' in locals() and scale_factor > 1 else None
        }
        if processed_scale is not None:
            result['processed_scale'] = processed_scale
        return result
    else:
        return final_cleaned_inverted

def save_debug_image(image, output_path, title=None, dpi_info=None, format=None, output_dpi=None):
    """
    Save a debug image with optional title using Pillow to preserve metadata

    Args:
        image: Image to save (grayscale or color)
        output_path: Path to save the image
        title: Optional title to add to the image
        dpi_info: Original DPI information
        format: Original image format to use
        output_dpi: User-selected DPI for output (overrides dpi_info if provided)
    """
    # Debug image info before processing
    debug_image_info(f"Before saving {os.path.basename(output_path)}", image)

    # Ensure image is in proper format for saving
    if image.dtype == bool:
        img = image.astype(np.uint8) * 255
    elif image.max() <= 1:
        img = (image * 255).astype(np.uint8)
    else:
        img = image.copy()

    # If grayscale, keep it as grayscale for PIL
    if len(img.shape) == 2:
        pil_mode = 'L'
    else:
        # Convert BGR to RGB for PIL (OpenCV uses BGR, PIL uses RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_mode = 'RGB'

    # Create PIL Image
    pil_img = Image.fromarray(img, mode=pil_mode)

    # Set DPI if provided (output_dpi takes precedence over dpi_info)
    if output_dpi:
        save_kwargs = {'dpi': output_dpi}
    elif dpi_info:
        save_kwargs = {'dpi': dpi_info}
    else:
        save_kwargs = {}

    # Determine format
    save_format = format if format else 'PNG'

    # If title is provided, create image with title; otherwise save clean image
    if title:
        # Create space for title
        title_height = 30

        if pil_mode == 'L':
            display_img = Image.new('RGB', (img.shape[1], img.shape[0] + title_height), color=(255, 255, 255))
            display_img.paste(pil_img, (0, title_height))
        else:
            display_img = Image.new('RGB', (img.shape[1], img.shape[0] + title_height), color=(255, 255, 255))
            display_img.paste(pil_img, (0, title_height))

        # Add title using OpenCV (PIL doesn't have good text support)
        display_np = np.array(display_img)
        cv2.putText(display_np, title, (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        display_img = Image.fromarray(display_np)

        # Save the titled version
        display_img.save(output_path, format=save_format, **save_kwargs)
    else:
        # Save clean image without title
        pil_img.save(output_path, format=save_format, **save_kwargs)

    print(f"Debug image saved to {output_path}")

def create_comparison_image(images, titles=None):
    """Create a comparison image grid using pure OpenCV"""
    # Check if all images are grayscale, convert to BGR if needed
    processed_images = []
    for img in images:
        if len(img.shape) == 2:
            # Convert grayscale to BGR
            processed_img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        else:
            processed_img = img.copy()
        processed_images.append(processed_img)

    # Get the maximum dimensions
    max_height = max(img.shape[0] for img in processed_images)
    max_width = max(img.shape[1] for img in processed_images)

    # Add padding and resize all images to the same size
    padded_images = []
    for img in processed_images:
        # Create a white canvas of the maximum size
        padded = np.ones((max_height, max_width, 3), dtype=np.uint8) * 255
        # Place the image in the center
        h, w = img.shape[:2]
        y_offset = (max_height - h) // 2
        x_offset = (max_width - w) // 2
        padded[y_offset:y_offset+h, x_offset:x_offset+w] = img
        padded_images.append(padded)

    # Define the grid layout (here: 2 rows, 3 columns)
    num_images = len(padded_images)
    rows = 2
    cols = 3

    # Add title space if titles are provided
    title_height = 30 if titles else 0

    # Create the comparison grid
    grid_height = rows * (max_height + title_height)
    grid_width = cols * max_width
    grid = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255

    # Place images in the grid
    for i, img in enumerate(padded_images):
        if i >= rows * cols:
            break

        row = i // cols
        col = i % cols

        y_start = row * (max_height + title_height)
        x_start = col * max_width

        # Add title if provided
        if titles and i < len(titles):
            cv2.putText(grid, titles[i], (x_start + 10, y_start + 20),
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        # Add image
        grid[y_start + title_height:y_start + title_height + max_height,
             x_start:x_start + max_width] = img

    return grid