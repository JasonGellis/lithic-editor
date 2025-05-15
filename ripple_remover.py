import cv2
import numpy as np
import networkx as nx
from skimage import morphology
from scipy.ndimage import label
import os
import traceback

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

print("Starting lithic_GUI.py")


def improve_line_quality_antialias(binary_image):
    """
    Improve line quality using anti-aliasing techniques

    Args:
        binary_image: Input binary image (black lines on white background)

    Returns:
        improved_image: Anti-aliased image with better line quality
    """
    # Ensure correct format
    if binary_image.max() <= 1:
        binary_image = binary_image * 255

    # Convert to proper format
    binary_image = binary_image.astype(np.uint8)

    # Scale up the image (4x)
    scale_factor = 4
    h, w = binary_image.shape
    upscaled = cv2.resize(binary_image, (w * scale_factor, h * scale_factor),
                         interpolation=cv2.INTER_CUBIC)

    # Apply slight Gaussian blur to smooth edges
    blurred = cv2.GaussianBlur(upscaled, (5, 5), 0.5)

    # Threshold back to binary with good contrast
    _, smoothed = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)

    # Scale back down to original size with anti-aliasing
    result = cv2.resize(smoothed, (w, h), interpolation=cv2.INTER_AREA)

    return result

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

def process_lithic_drawing_improved(image_path, output_folder="image_debug"):
    """
    Process a lithic drawing to remove ripple lines while preserving original line quality

    Args:
        image_path: Path to the input image
        output_folder: Folder to save all output images

    Returns:
        cleaned_image: Image with ripple lines removed but original line quality preserved
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    print(f"Created output folder: {output_folder}")

    # Add debugging header
    print("\n=== IMAGE DIMENSIONS DEBUGGING ===")
    print(f"{'Step':<30} {'Width':>10} {'Height':>10} {'Type':>15} {'Min':>8} {'Max':>8}")
    print("-" * 80)

    # Step 1: Read the image
    print("Reading image...")
    if isinstance(image_path, str):
        original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if original_image is None:
            raise ValueError(f"Could not read image at {image_path}")
        print(f"Image read successfully. Shape: {original_image.shape}")
    else:
        # Assume image_path is already a numpy array
        original_image = image_path
        print(f"Using provided numpy array. Shape: {original_image.shape}")

    # Save the original image
    save_debug_image(original_image, os.path.join(output_folder, '1_original_image.png'), 'Original Image')

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

    # Skeletonize the binary image to get thin lines
    print("Skeletonizing image...")
    skeleton = morphology.skeletonize(binary)
    skeleton_img = skeleton.astype(np.uint8) * 255
    print(f"Skeleton created. Non-zero pixels: {np.count_nonzero(skeleton)}")

    # Save the skeleton image
    save_debug_image(skeleton_img, os.path.join(output_folder, '2_skeleton.png'), 'Skeleton')

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

    save_debug_image(debug_img, os.path.join(output_folder, '3_endpoints_junctions.png'),
                    'Skeleton with Endpoints (Red) and Junctions (Green)')

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

    save_debug_image(segment_colors, os.path.join(output_folder, '4_labeled_segments.png'),
                    f'Labeled Segments (Total: {num_segments})')

    # Step 5: Build graph representation
    print("Building graph representation...")
    G = nx.Graph()

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

    save_debug_image(ripple_viz, os.path.join(output_folder, '5_ripple_identification.png'),
                    'Ripple Segments (Red) vs. Structural Lines (White)')

    # Step 7: Create mask of structural elements
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

    # Create a skeleton-based cleaned image for comparison
    skeleton_cleaned = np.zeros_like(skeleton_img)
    skeleton_cleaned[structural_mask] = 255

    # INVERT the skeleton-based cleaned image (black lines on white background)
    skeleton_cleaned_inverted = 255 - skeleton_cleaned

    # Save the skeleton-based cleaned image (inverted)
    save_debug_image(skeleton_cleaned_inverted, os.path.join(output_folder, '6_skeleton_cleaned.png'),
                    'Skeleton Cleaned')

    # Step 8: Create final image with original line quality preserved
    print("Creating final image with original line quality...")

    # Dilate the structural mask slightly to ensure we capture the full line thickness
    dilated_mask = morphology.binary_dilation(structural_mask, np.ones((2, 2), dtype=np.uint8))

    # Create the final cleaned image by using original binary pixels where they align with our dilated structure
    final_cleaned = np.zeros_like(binary_image)
    final_cleaned[dilated_mask & (binary_image > 0)] = 255

    # INVERT the final cleaned image (black lines on white background)
    final_cleaned_inverted = 255 - final_cleaned

    # Save the final cleaned image (inverted)
    save_debug_image(final_cleaned_inverted, os.path.join(output_folder, '7_final_cleaned.png'),
                    'Final Cleaned (Original Quality)')

    # Step 9: Apply line quality improvement
    print("Improving line quality...")
    improved_image = improve_line_quality_antialias(final_cleaned_inverted)

    # Save the improved quality image
    save_debug_image(improved_image, os.path.join(output_folder, '8_improved_quality.png'),
                    'Improved Line Quality')

    # Step 10: Export high-quality version (without title banner)
    print("Exporting high-quality image...")
    high_quality_path = os.path.join(output_folder, '9_high_quality.png')
    cv2.imwrite(high_quality_path, improved_image)
    print(f"High-quality PNG saved to {high_quality_path} with shape {improved_image.shape}")

    # Create comparison visualization with all versions
    comparison_image = create_comparison_image(
        [original_image, 255 - skeleton_img, skeleton_cleaned_inverted,
         final_cleaned_inverted, improved_image],
        ['Original Image', 'Original Skeleton', 'Skeleton Ripple Removed',
         'Original-Quality Ripple Removed', 'Improved Line Quality']
    )

    save_debug_image(comparison_image, os.path.join(output_folder, '10_comparison_all.png'))

    print("Processing complete!")
    return improved_image  # Return the improved version

def save_debug_image(image, output_path, title=None):
    """
    Save a debug image with optional title using OpenCV

    Args:
        image: Image to save (grayscale or color)
        output_path: Path to save the image
        title: Optional title to add to the image
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

    # If grayscale, convert to BGR for consistent handling
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Save the original image (without title banner)
    # This preserves the original dimensions
    cv2.imwrite(output_path, img)

    # Create a display version with title if needed
    if title:
        # Create space for title
        title_height = 30
        display_img = np.ones((img.shape[0] + title_height, img.shape[1], 3), dtype=np.uint8) * 255
        # Add image
        display_img[title_height:, :, :] = img
        # Add title
        cv2.putText(display_img, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

        # Save the display version with "_display" suffix
        display_path = output_path.replace('.png', '_display.png')
        cv2.imwrite(display_path, display_img)

        # Debug image info after processing (for the display version)
        debug_image_info(f"After processing {os.path.basename(display_path)}", display_img)
    else:
        # Debug image info after processing (for the original version)
        debug_image_info(f"After processing {os.path.basename(output_path)}", img)

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
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        # Add image
        grid[y_start + title_height:y_start + title_height + max_height,
             x_start:x_start + max_width] = img

    return grid