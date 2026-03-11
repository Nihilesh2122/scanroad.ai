"""
Flask Web Application for Smart Road Monitoring System
======================================================

This is the main web application that:
- Serves web pages (HTML templates)
- Handles image uploads
- Runs pothole detection
- Manages database operations
- Provides API endpoints

Author: AI Assistant
"""

# Import required libraries
from flask import Flask, render_template, request, redirect, url_for, flash, send_file
import os
import cv2
import numpy as np
from datetime import datetime

# Import our custom modules
import database
import pothole_detection

# Create Flask application instance
# This is the main entry point for our web application
app = Flask(__name__)

# Configuration settings
# SECRET_KEY is used for session management and security
app.secret_key = 'road-monitoring-secret-key-2024'

# Configure upload folders
# BASE_DIR: Project root directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# UPLOAD_FOLDER: Where uploaded images are stored
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')

# PROCESSED_FOLDER: Where processed images are stored
PROCESSED_FOLDER = os.path.join(BASE_DIR, 'static', 'processed')

# ALLOWED_EXTENSIONS: What file types are allowed for upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Apply configurations to Flask app
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER


def allowed_file(filename):
    """
    Check if the uploaded file has an allowed extension.
    
    Args:
        filename (str): Name of the file to check
    
    Returns:
        bool: True if extension is allowed, False otherwise
    """
    # Get the part after the last dot (extension)
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def ensure_directories():
    """
    Create necessary directories if they don't exist.
    """
    # Create uploads directory
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    # Create processed images directory
    os.makedirs(PROCESSED_FOLDER, exist_ok=True)
    
    # Create database
    database.init_db()


# ============================================================
# ROUTE: Home Page / Dashboard
# ============================================================
@app.route('/')
def index():
    """
    Render the main dashboard page.
    
    This is the home page that shows:
    - Navigation
    - Statistics summary
    - Recent pothole detections
    
    Returns:
        HTML: The dashboard template
    """
    try:
        # Get all potholes from database
        potholes = database.get_all_potholes()
        
        # Get statistics
        stats = database.get_pothole_stats()
        
        # Render the dashboard template with data
        return render_template('index.html', 
                             potholes=potholes, 
                             stats=stats)
    except Exception as e:
        # If error, show message and empty data
        flash(f'Error loading dashboard: {str(e)}', 'danger')
        return render_template('index.html', 
                             potholes=[], 
                             stats={'total': 0, 'low': 0, 'medium': 0, 'high': 0})


# ============================================================
# ROUTE: Upload and Detect Page
# ============================================================
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """
    Handle image upload and pothole detection.
    
    GET request: Show the upload form
    POST request: Process the uploaded image
    
    Returns:
        HTML: Upload form or results page
    """
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            flash('No file selected', 'danger')
            return redirect(request.url)
        
        file = request.files['file']
        
        # Check if a file was actually selected
        if file.filename == '':
            flash('No file selected', 'danger')
            return redirect(request.url)
        
        # Check if file type is allowed
        if file and allowed_file(file.filename):
            try:
                # Save the uploaded file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"road_{timestamp}_{file.filename}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Run pothole detection
                result = pothole_detection.detect_potholes(filepath, PROCESSED_FOLDER)
                
                if result['success']:
                    # Save the processed image path (relative for web)
                    processed_filename = os.path.basename(result['output_path'])
                    processed_path = f"processed/{processed_filename}"
                    
                    # Get location from form (or use default)
                    location = request.form.get('location', 'Main Road')
                    
                    # If potholes detected, save to database
                    if result['count'] > 0:
                        # Determine overall severity (worst one)
                        overall_severity = max(result['severities'], 
                                             key=lambda x: {'Low': 1, 'Medium': 2, 'High': 3}[x])
                        
                        # Insert into database
                        pothole_id = database.insert_pothole(
                            location=location,
                            severity=overall_severity,
                            image_path=processed_path
                        )
                        
                        flash(f'Detected {result["count"]} pothole(s)!', 'success')
                    else:
                        flash('No potholes detected in this image.', 'info')
                        pothole_id = None
                    
                    # Show results page
                    return render_template('result.html',
                                         original_image=f"uploads/{filename}",
                                         processed_image=processed_path,
                                         pothole_count=result['count'],
                                         severities=result['severities'],
                                         location=location)
                else:
                    flash(f'Error processing image: {result.get("error", "Unknown error")}', 'danger')
                    
            except Exception as e:
                flash(f'Error processing image: {str(e)}', 'danger')
    
    # GET request - show upload form
    return render_template('upload.html')


# ============================================================
# ROUTE: Webcam Detection
# ============================================================
@app.route('/webcam')
def webcam():
    """
    Render the webcam detection page.
    
    This page shows real-time pothole detection using webcam.
    
    Returns:
        HTML: Webcam template
    """
    return render_template('webcam.html')


# ============================================================
# ROUTE: API - Process Webcam Frame
# ============================================================
@app.route('/process_frame', methods=['POST'])
def process_frame():
    """
    Process a frame from webcam for pothole detection.
    
    This is an AJAX endpoint that receives an image from
    the webcam and returns detection results.
    
    Returns:
        JSON: Detection results
    """
    try:
        # Get the image data from the request
        if 'frame' not in request.files:
            return {'success': False, 'error': 'No frame received'}
        
        file = request.files['frame']
        
        # Read the image as numpy array
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Run pothole detection
        result = pothole_detection.detect_potholes_from_array(image)
        
        if result['success']:
            # Encode the result image to send back
            _, buffer = cv2.imencode('.jpg', result['image'])
            import base64
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return {
                'success': True,
                'image': f'data:image/jpeg;base64,{image_base64}',
                'count': result['count'],
                'severities': result['severities']
            }
        else:
            return {'success': False, 'error': result.get('error', 'Unknown error')}
            
    except Exception as e:
        return {'success': False, 'error': str(e)}


# ============================================================
# ROUTE: Delete Pothole Record
# ============================================================
@app.route('/delete/<int:pothole_id>')
def delete_pothole(pothole_id):
    """
    Delete a pothole record from the database.
    
    Args:
        pothole_id (int): ID of the pothole to delete
    
    Returns:
        Redirect: Back to the dashboard
    """
    try:
        # Delete from database
        deleted = database.delete_pothole(pothole_id)
        
        if deleted:
            flash('Pothole record deleted successfully!', 'success')
        else:
            flash('Could not delete the record.', 'warning')
            
    except Exception as e:
        flash(f'Error deleting record: {str(e)}', 'danger')
    
    # Redirect back to dashboard
    return redirect(url_for('index'))


# ============================================================
# ROUTE: About Page
# ============================================================
@app.route('/about')
def about():
    """
    Render the about page with project information.
    
    Returns:
        HTML: About template
    """
    return render_template('about.html')


# ============================================================
# Main Entry Point
# ============================================================
if __name__ == '__main__':
    """
    Run the Flask application when this file is executed directly.
    """
    # Ensure directories exist
    ensure_directories()
    
    print("=" * 50)
    print("Smart Road Monitoring System")
    print("=" * 50)
    print("Starting Flask server...")
    print("Open your browser and go to: http://127.0.0.1:5000")
    print("=" * 50)
    
    # Run the Flask development server
    # debug=True enables auto-reload and helpful error messages
    app.run(debug=True, host='0.0.0.0', port=5000)

