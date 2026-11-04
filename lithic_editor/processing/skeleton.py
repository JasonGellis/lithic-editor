"""
The skeleton of the drawing: thinning, spur removal, endpoints and junctions.
"""

import logging

import cv2
import numpy as np
from scipy.ndimage import convolve
from skimage.morphology import skeletonize

logger = logging.getLogger(__name__)

_EIGHT_NEIGHBOURS = np.ones((3, 3), dtype=np.int32)


def make_skeleton(ink_image, spur_length_px):
    """
    Thin the ink to a one-pixel skeleton, smooth it and remove short spurs.

    ``ink_image`` is a uint8 image with 255 for ink. Returns a boolean skeleton.
    """
    logger.info("Thinning structural elements using Lee skeletonization...")
    skeleton = skeletonize(ink_image > 0, method='lee').astype(bool)
    logger.info("Using Lee skeletonization method")
    skeleton_img = skeleton.astype(np.uint8) * 255
    logger.info(f"Skeleton created. Non-zero pixels: {np.count_nonzero(skeleton)}")

    # Light dilation and closing join small gaps, then the skeleton is thinned again
    logger.info("Applying post-skeleton morphological operations...")
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    smoothed = cv2.dilate(skeleton_img, dilate_kernel, iterations=1)
    logger.info("Applied dilation to thicken skeleton slightly")
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_CLOSE, close_kernel)
    logger.info("Applied closing to connect skeleton gaps")
    skeleton = skeletonize(smoothed > 0, method='lee').astype(bool)
    logger.info("Using Lee method for re-thinning")
    logger.info(f"Smoothed skeleton created. Non-zero pixels: {np.count_nonzero(skeleton)}")

    logger.info("Analyzing branch lengths to remove spurs...")
    pruned = remove_short_branches(skeleton, min_length=spur_length_px)
    removed = np.count_nonzero(skeleton) - np.count_nonzero(pruned)
    logger.info(f"Branch length analysis removed {removed} pixels")
    logger.info(f"Final skeleton created. Non-zero pixels: {np.count_nonzero(pruned)}")
    return pruned


def remove_short_branches(skel, min_length=5):
    """Remove branches shorter than min_length that connect to junctions"""
    skel_copy = skel.copy()
    h, w = skel.shape

    # Find endpoints (interior pixels with exactly 1 neighbor), in row-major order
    neighbor_count = convolve(skel.astype(np.int32), _EIGHT_NEIGHBOURS, mode='constant') - skel
    is_endpoint = skel & (neighbor_count == 1)
    is_endpoint[0, :] = is_endpoint[-1, :] = False
    is_endpoint[:, 0] = is_endpoint[:, -1] = False
    endpoints = [(int(x), int(y)) for y, x in np.argwhere(is_endpoint)]

    logger.info(f"Found {len(endpoints)} endpoints to analyze")

    # For each endpoint, trace back to find branch length
    branches_to_remove = []

    for ex, ey in endpoints:
        # Trace from endpoint until we hit a junction or another endpoint
        path = [(ex, ey)]
        current_x, current_y = ex, ey
        visited = set()
        visited.add((current_x, current_y))

        while True:
            # Find next pixel in the path
            next_pixels = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = current_x + dx, current_y + dy
                    if (0 <= nx < w and 0 <= ny < h and
                        skel[ny, nx] and (nx, ny) not in visited):
                        next_pixels.append((nx, ny))

            if len(next_pixels) == 0:
                # Dead end (shouldn't happen in a connected skeleton)
                break
            elif len(next_pixels) == 1:
                # Continue along the branch
                current_x, current_y = next_pixels[0]
                path.append((current_x, current_y))
                visited.add((current_x, current_y))
            else:
                # We've reached a junction (multiple possible next pixels)
                break

            # Safety check to prevent infinite loops
            if len(path) > 100:
                break

        # Check if this branch should be removed
        branch_length = len(path)

        # Only remove if:
        # 1. Branch is shorter than min_length
        # 2. Branch ends at a junction (not isolated)
        if branch_length <= min_length and len(path) > 1:
            # Check if the end of the path is a junction
            end_x, end_y = path[-1]
            if end_x != ex or end_y != ey:  # Make sure we actually traced somewhere
                end_neighbors = np.sum(skel[end_y-1:end_y+2, end_x-1:end_x+2]) - 1
                if end_neighbors >= 3:  # It's a junction
                    # Mark this branch for removal (except the junction point)
                    branches_to_remove.extend(path[:-1])
                    logger.info(f"Marking branch of length {branch_length} for removal")

    # Remove the marked branches
    for x, y in branches_to_remove:
        skel_copy[y, x] = False

    return skel_copy


def find_endpoints_and_junctions(skeleton):
    """Interior skeleton pixels with exactly one 8-neighbour, and with three or more."""
    neighbor_count = convolve(skeleton.astype(np.int32), _EIGHT_NEIGHBOURS, mode='constant') - skeleton
    interior = np.zeros_like(skeleton)
    interior[1:-1, 1:-1] = True
    endpoints = [(int(x), int(y)) for y, x in np.argwhere(skeleton & interior & (neighbor_count == 1))]
    junctions = [(int(x), int(y)) for y, x in np.argwhere(skeleton & interior & (neighbor_count >= 3))]
    return endpoints, junctions


def bridge_close_endpoints(skeleton, endpoints, max_distance):
    """
    Draw a one-pixel line between every pair of free ends within ``max_distance``.

    Modifies ``skeleton`` in place and returns the number of bridges drawn. Each
    endpoint is bridged at most once, to its nearest partner.
    """
    if len(endpoints) < 2 or max_distance <= 0:
        return 0
    points = np.array(endpoints, dtype=np.float64)
    used = set()
    bridges = 0
    canvas = skeleton.view(np.uint8)
    for i in range(len(points)):
        if i in used:
            continue
        distances = np.hypot(*(points - points[i]).T)
        distances[i] = np.inf
        for j in used:
            distances[j] = np.inf
        j = int(np.argmin(distances))
        if distances[j] <= max_distance:
            cv2.line(canvas, tuple(int(v) for v in points[i]), tuple(int(v) for v in points[j]), 1, 1)
            used.update((i, j))
            bridges += 1
    return bridges


def find_special_points(skeleton, bridge_distance, y_tip_threshold):
    """
    Find the endpoints and junctions of ``skeleton``.

    Free ends closer together than ``bridge_distance`` are first joined, in place:
    thinning can leave a break of a few pixels in a contour, and an open contour
    would be classed as a ripple. Then every junction within ``y_tip_threshold`` of
    an endpoint becomes an endpoint itself, so a Y-shaped ripple tip counts as part
    of the ripple. Returns ``(endpoints, junctions)`` as lists of ``(x, y)``.
    """
    logger.info("Finding endpoints and junctions...")
    logger.info(f"Total skeleton pixels: {np.count_nonzero(skeleton)}")
    endpoints, junctions = find_endpoints_and_junctions(skeleton)
    bridged = bridge_close_endpoints(skeleton, endpoints, bridge_distance)
    if bridged:
        logger.info(f"Bridged {bridged} skeleton gaps of up to {bridge_distance:.1f} px")
        endpoints, junctions = find_endpoints_and_junctions(skeleton)
    logger.info(f"Found {len(endpoints)} endpoints and {len(junctions)} junctions")

    logger.info("Converting Y-tip junctions to endpoints...")
    logger.info(f"Using Y-tip threshold: {y_tip_threshold} pixels")
    converted = []
    for jx, jy in junctions:
        min_distance = float('inf')
        for ex, ey in endpoints:
            distance = ((jx - ex) ** 2 + (jy - ey) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
                if distance <= y_tip_threshold:
                    break
        if min_distance <= y_tip_threshold:
            converted.append((jx, jy))
            logger.info(f"  Converting junction at ({jx},{jy}) to endpoint "
                        f"(distance {min_distance:.1f} to nearest endpoint)")
    for junction in converted:
        junctions.remove(junction)
        endpoints.append(junction)
    logger.info(f"Converted {len(converted)} Y-tip junctions to endpoints")
    logger.info(f"Updated: {len(endpoints)} endpoints and {len(junctions)} junctions")
    return endpoints, junctions


def special_points_picture(skeleton, endpoints, junctions):
    """The skeleton in white with endpoints in red and junctions in green (BGR)."""
    height, width = skeleton.shape
    picture = np.zeros((height, width, 3), dtype=np.uint8)
    picture[skeleton] = [255, 255, 255]
    for x, y in endpoints:
        cv2.circle(picture, (x, y), 1, (0, 0, 255), -1)
    for x, y in junctions:
        cv2.circle(picture, (x, y), 1, (0, 255, 0), -1)
    return picture
