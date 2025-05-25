# Configuration file for Multi-Face Attendance System (SQLite Version)
# Optimized for Lenovo Legion 5i 2022 i7 12th Gen

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database Configuration - Using SQLite for easy setup
DATABASE_URL = "sqlite:///attendance_system.db"

# Face Recognition Configuration (Optimized for i7 12th Gen)
FACE_RECOGNITION_CONFIG = {
    'tolerance': 0.6,
    'model': 'hog',  # Faster on CPU
    'min_confidence': 0.6,
    'max_faces_per_frame': 70,  # Support for 70 faces
    'face_detection_batch_size': 8,  # Optimized for laptop hardware
    'encoding_cache_size': 1000,  # Cache encodings in memory
    'use_gpu': False,  # Set to True if GPU available
    'num_jitters': 1,  # Lower for speed, higher for accuracy
    'resize_factor': 0.25  # Resize frame for faster processing
}

# Performance Optimization (Laptop-specific)
PERFORMANCE_CONFIG = {
    'max_workers': 8,  # Utilize i7 cores efficiently
    'memory_limit_mb': 4096,  # 4GB memory limit
    'image_quality': 85,  # Balance quality vs storage
    'max_image_size_mb': 5,  # Max upload size
    'cache_ttl_seconds': 3600,  # 1 hour cache
    'enable_logging': True,
    'log_level': 'INFO'
}

# Authentication Configuration
AUTH_CONFIG = {
    'session_timeout_hours': 8,
    'max_login_attempts': 3,
    'password_min_length': 8,
    'require_special_chars': True,
    'jwt_secret_key': os.getenv('JWT_SECRET', 'your-secret-key-change-this'),
    'cookie_expiry_days': 7
}

# Email Configuration
EMAIL_CONFIG = {
    'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
    'smtp_port': int(os.getenv('SMTP_PORT', '587')),
    'email': os.getenv('EMAIL_USER', 'your-email@gmail.com'),
    'password': os.getenv('EMAIL_PASSWORD', 'your-app-password'),
    'use_tls': True
}

# Notification Settings
NOTIFICATION_CONFIG = {
    'email_notifications': True,
    'attendance_threshold': 75,  # Alert if below 75%
    'late_threshold_minutes': 15,
    'absent_notification_delay_hours': 2
}

# File Paths
PHOTOS_DIR = "data/photos"
MODELS_DIR = "models"
EXPORTS_DIR = "data/exports"
LOGS_DIR = "logs"
TEMP_DIR = "data/temp"
BACKUP_DIR = "data/backups"

# Academic Configuration
ACADEMIC_CONFIG = {
    'academic_year_start_month': 7,  # July
    'semester_duration_months': 6,
    'default_class_duration_minutes': 60,
    'attendance_grace_period_minutes': 10
}

# Security Configuration
SECURITY_CONFIG = {
    'encrypt_face_data': True,
    'secure_file_storage': True,
    'audit_logs': True,
    'max_file_uploads_per_hour': 100,
    'allowed_image_types': ['jpg', 'jpeg', 'png', 'bmp'],
    'scan_uploads_for_malware': False  # Set to True if antivirus available
}

# System Monitoring
MONITORING_CONFIG = {
    'enable_performance_monitoring': True,
    'cpu_threshold': 80,  # Alert if CPU > 80%
    'memory_threshold': 85,  # Alert if Memory > 85%
    'disk_threshold': 90,  # Alert if Disk > 90%
    'monitor_interval_seconds': 60
}

# UI Configuration
UI_CONFIG = {
    'theme': 'light',  # 'light' or 'dark'
    'primary_color': '#1f77b4',
    'sidebar_expanded': True,
    'show_tips': True,
    'animation_enabled': True,
    'charts_theme': 'plotly'
}

# Create directories if they don't exist
directories = [PHOTOS_DIR, MODELS_DIR, EXPORTS_DIR, LOGS_DIR, TEMP_DIR, BACKUP_DIR]
for directory in directories:
    os.makedirs(directory, exist_ok=True)

# Default Admin User (Change this!)
DEFAULT_ADMIN = {
    'username': 'admin',
    'password': 'admin123',  # Change this immediately!
    'email': 'admin@school.edu',
    'role': 'admin'
} 