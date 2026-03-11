"""
Pothole Detection Module for Smart Road Monitoring System
==========================================================

This module handles all image processing and computer vision operations
to detect potholes in road images using OpenCV.

The detection process:
1. Accept road image
2. Convert to grayscale
3. Apply edge detection (Canny)
4. Find contours
5. Identify pothole-like regions
6. Draw bounding boxes
7. Classify severity

Author: AI Assistant
"""

import cv2
import numpy as np
import os
from datetime import datetime


def preprocess_image(image):
    """
    Preprocess the image to make it suitable for pothole detection.
    
    Steps:
    1. Resize image to consistent size (for faster processing)
    2. Convert to grayscale
    3. Apply Gaussian blur to reduce noise
    4. Apply histogram equalization to improve contrast
    
    Args:
        image: Original image (color or grayscale)
    
    Returns:
        tuple: (grayscale_image, original_image_copy)
    """
    # Get original dimensions
    height, width = image.shape[:2]
    
    # Resize for consistent processing (max 800px width)
    max_width = 800
    if width > max_width:
        scale = max_width / width
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height))
    
    # Make a copy of the original image
    original = image.copy()
    
    # Convert to grayscale (black and white)
    # This simplifies the image and makes edge detection easier
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise and smooth the image
    # Kernel size of 5x5 helps reduce false edges
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply histogram equalization to improve contrast
    # This makes dark and light areas more distinguishable
    equalized = cv2.equalizeHist(blurred)
    
    return equalized, original


def detect_edges(gray_image):
    """
    Detect edges in the grayscale image using Canny edge detection.
    
    Canny edge detection is a multi-stage algorithm:
    1. Noise reduction
    2. Gradient calculation
    3. Non-maximum suppression
    4. Double threshold
    5. Edge tracking by hysteresis
    
    Args:
        gray_image: Grayscale image
    
    Returns:
        numpy.ndarray: Binary image with detected edges
    """
    # Apply Canny edge detection
    # First value (50) is lower threshold - detects weak edges
    # Second value (150) is upper threshold - detects strong edges
    edges = cv2.Canny(gray_image, 50, 150)
    
    return edges


def find_pothole_contours(edges):
    """
    Find contours (closed shapes) in the edge-detected image.
    
    Contours are curves joining all continuous points having the 
    same color or intensity. Potholes appear as dark enclosed regions.
    
    Args:
        edges: Edge-detected binary image
    
    Returns:
        list: List of contours found in the image
    """
    # Find contours in the edge image
    # cv2.RETR_EXTERNAL: Only retrieves external contours
    # cv2.CHAIN_APPROX_SIMPLE: Saves memory by compressing horizontal,
    #                         vertical, and diagonal segments
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours


def filter_potholes(contours, min_area=500, max_area=50000):
    """
    Filter contours to identify potential potholes based on their properties.
    
    Potholes typically:
    - Are enclosed (not open lines)
    - Have area between min and max thresholds
    - Are somewhat circular or irregular (not long rectangles)
    
    Args:
        contours: List of all detected contours
        min_area: Minimum area to consider (filters noise)
        max_area: Maximum area (filters very large regions)
    
    Returns:
        list: Filtered list of potential pothole contours
    """
    potholes = []
    
    for contour in contours:
        # Calculate the area of the contour
        area = cv2.contourArea(contour)
        
        # Skip very small areas (likely noise)
        if area < min_area:
            continue
        
        # Skip very large areas (likely not potholes)
        if area > max_area:
            continue
        
        # Get the perimeter of the contour
        perimeter = cv2.arcLength(contour, True)
        
        # Calculate circularity - how circular is the shape?
        # Circularity = 4 * pi * area / perimeter^2
        # A perfect circle has circularity = 1
        if perimeter > 0:
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            
            # Potholes are usually somewhat irregular but not too elongated
            # Accept circularity between 0.1 and 0.8
            if 0.1 < circularity < 0.8:
                potholes.append(contour)
    
    return potholes


def classify_severity(contour):
    """
    Classify the severity of a pothole based on its characteristics.
    
    Severity is determined by:
    - Area: Larger potholes = more severe
    - Depth (approximated by darkness): Darker regions = deeper potholes
    
    Args:
        contour: The contour of the pothole
    
    Returns:
        str: Severity level - 'Low', 'Medium', or 'High'
    """
    # Calculate area
    area = cv2.contourArea(contour)
    
    # Classify based on area size
    if area < 2000:
        return 'Low'
    elif area < 8000:
        return 'Medium'
    else:
        return 'High'


def draw_pothole_boxes(image, potholes):
    """
    Draw bounding boxes around detected potholes with severity labels.
    
    Args:
        image: Original image to draw on
        potholes: List of pothole contours
    
    Returns:
        tuple: (annotated_image, severity_list)
    """
    # Make a copy to avoid modifying original
    result = image.copy()
    
    # List to store severity of each detected pothole
    severities = []
    
    # Define colors for each severity level (BGR format for OpenCV)
    colors = {
        'Low': (0, 255, 0),      # Green
        'Medium': (0, 165, 255), # Orange
        'High': (0, 0, 255)      # Red
    }
    
    for i, contour in enumerate(potholes):
        # Get the bounding rectangle for this contour
        x, y, w, h = cv2.boundingRect(contour)
        
        # Classify severity
        severity = classify_severity(contour)
        severities.append(severity)
        
        # Get the color for this severity
        color = colors[severity]
        
        # Draw rectangle around pothole
        cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
        
        # Add label with severity
        label = f"Pothole #{i+1}: {severity}"
        cv2.putText(result, label, (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw contour outline (optional - helps visualize the shape)
        cv2.drawContours(result, [contour], -1, color, 1)
    
    return result, severities


def detect_potholes(image_path, output_dir=None):
    """
    Main function to detect potholes in a road image.
    
    This function orchestrates the entire detection pipeline:
    1. Load the image
    2. Preprocess (grayscale, blur, equalize)
    3. Detect edges
    4. Find and filter contours
    5. Draw bounding boxes
    6. Classify severity
    
    Args:
        image_path (str): Path to the input road image
        output_dir (str): Directory to save processed images (optional)
    
    Returns:
        dict: Results containing:
            - 'success': Boolean indicating if detection was successful
            - 'image': Annotated image with bounding boxes
            - 'count': Number of potholes detected
            - 'severities': List of severity levels
            - 'output_path': Path to saved image (if saved)
    """
    # Load the image from file
    # cv2.IMREAD_COLOR loads the image in color (3 channels)
    image = cv2.imread(image_path)
    
    # Check if image was loaded successfully
    if image is None:
        return {
            'success': False,
            'error': f"Could not load image from {image_path}"
        }
    
    # Step 1: Preprocess the image
    gray, original = preprocess_image(image)
    
    # Step 2: Detect edges
    edges = detect_edges(gray)
    
    # Step 3: Find contours
    contours = find_pothole_contours(edges)
    
    # Step 4: Filter to find potential potholes
    potholes = filter_potholes(contours)
    
    # Step 5: Draw bounding boxes and get severity
    result_image, severities = draw_pothole_boxes(original, potholes)
    
    # Step 6: Save the processed image if output directory specified
    output_path = None
    if output_dir:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"detected_{timestamp}.jpg"
        output_path = os.path.join(output_dir, filename)
        
        # Save the image
        cv2.imwrite(output_path, result_image)
    
    # Return successful result
    return {
        'success': True,
        'image': result_image,
        'count': len(potholes),
        'severities': severities,
        'output_path': output_path
    }


def detect_potholes_from_array(image_array):
    """
    Detect potholes from an image array (for webcam frames).
    
    This is similar to detect_potholes but accepts a numpy array
    instead of a file path. Used for real-time webcam detection.
    
    Args:
        image_array: NumPy array representing the image
    
    Returns:
        dict: Detection results (same as detect_potholes)
    """
    # Check if valid image
    if image_array is None:
        return {
            'success': False,
            'error': "Invalid image array"
        }
    
    # Preprocess
    gray, original = preprocess_image(image_array)
    
    # Detect edges
    edges = detect_edges(gray)
    
    # Find contours
    contours = find_pothole_contours(edges)
    
    # Filter potholes
    potholes = filter_potholes(contours)
    
    # Draw boxes
    result_image, severities = draw_pothole_boxes(original, potholes)
    
    return {
        'success': True,
        'image': result_image,
        'count': len(potholes),
        'severities': severities
    }


# Test the module when run directly
if __name__ == '__main__':
    print("Pothole Detection Module")
    print("=========================")
    print("This module provides pothole detection functionality.")
    print("Import it in your Flask app to use the detection features.")

