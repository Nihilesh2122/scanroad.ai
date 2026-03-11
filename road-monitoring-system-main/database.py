"""
Database Module for Smart Road Monitoring System
=================================================

This file handles all database operations using SQLite.
It creates the potholes table and provides functions to:
- Insert new pothole detections
- Retrieve pothole history
- Get statistics

Author: AI Assistant
"""

import sqlite3
from datetime import datetime
import os

# Get the absolute path to the database file
# This ensures the database is stored in the project root
DATABASE_PATH = os.path.join(os.path.dirname(__file__), 'potholes.db')


def get_db_connection():
    """
    Create and return a database connection.
    
    Returns:
        sqlite3.Connection: A connection to the SQLite database
    """
    # Connect to the SQLite database (creates it if it doesn't exist)
    conn = sqlite3.connect(DATABASE_PATH)
    
    # Return rows as dictionaries instead of tuples (easier to work with)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """
    Initialize the database by creating the potholes table if it doesn't exist.
    
    The potholes table stores:
    - id: Unique identifier (auto-incremented)
    - location: Where the pothole was detected
    - severity: How severe the damage is (Low, Medium, High)
    - image_path: Path to the stored image
    - date_detected: When the pothole was detected
    """
    # Get database connection
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create the potholes table with specified columns
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS potholes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            location TEXT NOT NULL,
            severity TEXT NOT NULL,
            image_path TEXT NOT NULL,
            date_detected TEXT NOT NULL
        )
    ''')
    
    # Commit the changes and close the connection
    conn.commit()
    conn.close()
    
    print("Database initialized successfully!")


def insert_pothole(location, severity, image_path):
    """
    Insert a new pothole detection into the database.
    
    Args:
        location (str): Description of where the pothole was detected
        severity (str): Severity level - 'Low', 'Medium', or 'High'
        image_path (str): Path to the processed image file
    
    Returns:
        int: The ID of the newly inserted pothole record
    """
    # Get current date and time
    date_detected = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Connect to database
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Insert the new pothole record
    cursor.execute('''
        INSERT INTO potholes (location, severity, image_path, date_detected)
        VALUES (?, ?, ?, ?)
    ''', (location, severity, image_path, date_detected))
    
    # Get the ID of the inserted record
    pothole_id = cursor.lastrowid
    
    # Save changes and close connection
    conn.commit()
    conn.close()
    
    return pothole_id


def get_all_potholes():
    """
    Retrieve all pothole records from the database.
    
    Returns:
        list: A list of dictionaries containing pothole data
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get all potholes ordered by date (newest first)
    cursor.execute('SELECT * FROM potholes ORDER BY date_detected DESC')
    
    # Fetch all rows and convert to list of dictionaries
    potholes = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    return potholes


def get_pothole_stats():
    """
    Get statistics about pothole detections.
    
    Returns:
        dict: Dictionary containing counts of total and severity levels
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get total count
    cursor.execute('SELECT COUNT(*) as total FROM potholes')
    total = cursor.fetchone()['total']
    
    # Get count by severity
    cursor.execute('''
        SELECT severity, COUNT(*) as count 
        FROM potholes 
        GROUP BY severity
    ''')
    
    # Create a dictionary of severity counts
    severity_counts = {row['severity']: row['count'] for row in cursor.fetchall()}
    
    conn.close()
    
    # Return comprehensive statistics
    return {
        'total': total,
        'low': severity_counts.get('Low', 0),
        'medium': severity_counts.get('Medium', 0),
        'high': severity_counts.get('High', 0)
    }


def delete_pothole(pothole_id):
    """
    Delete a pothole record from the database.
    
    Args:
        pothole_id (int): The ID of the pothole to delete
    
    Returns:
        bool: True if deleted successfully, False otherwise
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Delete the pothole with the given ID
    cursor.execute('DELETE FROM potholes WHERE id = ?', (pothole_id,))
    
    # Check if any row was deleted
    deleted = cursor.rowcount > 0
    
    conn.commit()
    conn.close()
    
    return deleted


# This block runs when the file is executed directly (not imported)
if __name__ == '__main__':
    # Initialize the database
    init_db()
    print("Database setup complete!")

