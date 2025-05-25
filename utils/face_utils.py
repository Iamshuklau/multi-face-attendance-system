import cv2
import face_recognition
import numpy as np
import pickle
from PIL import Image
import os
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional, Dict
import psutil
import sys
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FACE_RECOGNITION_CONFIG, PERFORMANCE_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceRecognitionCache:
    """In-memory cache for face encodings to improve performance."""
    
    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size
        self.access_times = {}
    
    def get(self, key):
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None
    
    def set(self, key, value):
        if len(self.cache) >= self.max_size:
            # Remove least recently used item
            oldest_key = min(self.access_times, key=self.access_times.get)
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        self.cache[key] = value
        self.access_times[key] = time.time()
    
    def clear(self):
        self.cache.clear()
        self.access_times.clear()

# Global cache instance
face_cache = FaceRecognitionCache()

def check_gpu_availability():
    """Check if GPU is available and configure accordingly."""
    try:
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            logger.info(f"GPU available: {gpu_name} (Count: {gpu_count})")
            return True, gpu_name
        else:
            logger.info("GPU not available, using CPU")
            return False, "CPU"
    except Exception as e:
        logger.warning(f"Error checking GPU: {e}")
        return False, "CPU"

def monitor_system_resources():
    """Monitor current system resource usage."""
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_available_mb = memory.available / (1024 * 1024)
        
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'memory_available_mb': memory_available_mb
        }
    except Exception as e:
        logger.error(f"Error monitoring resources: {e}")
        return {'cpu_percent': 0, 'memory_percent': 0, 'memory_available_mb': 1000}

def optimize_image_for_processing(image, max_size=(800, 600)):
    """Optimize image size for faster processing while maintaining quality."""
    height, width = image.shape[:2]
    
    # Calculate scaling factor
    scale_x = max_size[0] / width
    scale_y = max_size[1] / height
    scale = min(scale_x, scale_y, 1.0)  # Don't upscale
    
    if scale < 1.0:
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return resized
    
    return image

def enhance_image_quality(image):
    """Enhance image quality for better face recognition."""
    try:
        # Convert to LAB color space for better processing
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels and convert back
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        return enhanced
    except Exception as e:
        logger.warning(f"Error enhancing image: {e}")
        return image

def extract_face_encoding(image_path, enhance_quality=True):
    """Extract face encoding from an image file with optimizations and proper type conversion."""
    try:
        # Check cache first
        cache_key = f"file_{image_path}_{os.path.getmtime(image_path)}"
        cached_encoding = face_cache.get(cache_key)
        if cached_encoding is not None:
            return cached_encoding
        
        start_time = time.time()
        
        # Load and optimize image
        image = face_recognition.load_image_file(image_path)
        
        if enhance_quality:
            image = enhance_image_quality(image)
        
        # Optimize for processing
        image = optimize_image_for_processing(image)
        
        # Find face locations with optimized model
        face_locations = face_recognition.face_locations(
            image, 
            model=FACE_RECOGNITION_CONFIG['model'],
            number_of_times_to_upsample=1
        )
        
        if len(face_locations) == 0:
            logger.warning(f"No faces detected in {image_path}")
            return None
        
        # Get face encodings
        face_encodings = face_recognition.face_encodings(
            image, 
            face_locations,
            num_jitters=FACE_RECOGNITION_CONFIG['num_jitters']
        )
        
        if len(face_encodings) > 0:
            encoding = face_encodings[0]
            
            # Calculate quality score and convert to Python float
            quality_score = calculate_face_quality(image, face_locations[0])
            quality_score = float(quality_score)  # Convert NumPy float to Python float
            
            # Cache the result
            face_cache.set(cache_key, (encoding, quality_score))
            
            processing_time = time.time() - start_time
            logger.info(f"Face encoding extracted in {processing_time:.2f}s, quality: {quality_score:.2f}")
            
            return encoding, float(quality_score)
        else:
            return None
            
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")
        return None

def calculate_face_quality(image, face_location):
    """Calculate face quality score based on various factors."""
    top, right, bottom, left = face_location
    face_image = image[top:bottom, left:right]
    
    if face_image.size == 0:
        return 0.0
    
    # Calculate sharpness using Laplacian variance
    gray_face = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
    sharpness = cv2.Laplacian(gray_face, cv2.CV_64F).var()
    
    # Calculate brightness
    brightness = np.mean(gray_face)
    
    # Calculate contrast
    contrast = gray_face.std()
    
    # Calculate size score
    face_area = (bottom - top) * (right - left)
    size_score = min(face_area / 10000, 1.0)  # Normalize to max 1.0
    
    # Combine scores (weighted average)
    quality_score = (
        min(sharpness / 100, 1.0) * 0.3 +  # Sharpness weight
        min(brightness / 255, 1.0) * 0.2 +  # Brightness weight
        min(contrast / 100, 1.0) * 0.2 +    # Contrast weight
        size_score * 0.3                     # Size weight
    )
    
    # Convert to Python float to avoid NumPy type issues
    return float(quality_score)

def detect_liveness(image, face_location):
    """Basic liveness detection to prevent photo spoofing."""
    top, right, bottom, left = face_location
    face_image = image[top:bottom, left:right]
    
    if face_image.size == 0:
        return False
    
    # Convert to grayscale
    gray_face = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
    
    # Calculate texture analysis using Local Binary Pattern variance
    # This helps detect flat surfaces (photos) vs real faces
    texture_variance = cv2.Laplacian(gray_face, cv2.CV_64F).var()
    
    # Simple threshold - real faces typically have higher texture variance
    liveness_threshold = 100  # Adjust based on testing
    
    return texture_variance > liveness_threshold

def extract_face_encoding_from_array(image_array, enable_liveness_check=False):
    """Extract face encoding from a numpy array with optimizations."""
    try:
        start_time = time.time()
        
        # Optimize image for processing
        optimized_image = optimize_image_for_processing(image_array)
        
        # Find face locations
        face_locations = face_recognition.face_locations(
            optimized_image,
            model=FACE_RECOGNITION_CONFIG['model']
        )
        
        if len(face_locations) == 0:
            return [], []
        
        # Limit number of faces for performance
        max_faces = FACE_RECOGNITION_CONFIG['max_faces_per_frame']
        if len(face_locations) > max_faces:
            # Sort by face size and take largest faces
            face_locations = sorted(face_locations, 
                                  key=lambda loc: (loc[2]-loc[0]) * (loc[1]-loc[3]), 
                                  reverse=True)[:max_faces]
        
        # Get face encodings
        face_encodings = face_recognition.face_encodings(
            optimized_image, 
            face_locations,
            num_jitters=FACE_RECOGNITION_CONFIG['num_jitters']
        )
        
        # Filter faces based on quality and liveness
        filtered_encodings = []
        filtered_locations = []
        
        for encoding, location in zip(face_encodings, face_locations):
            quality_score = calculate_face_quality(optimized_image, location)
            
            # Skip low quality faces
            if quality_score < 0.3:
                continue
            
            # Liveness check if enabled
            if enable_liveness_check:
                if not detect_liveness(optimized_image, location):
                    logger.warning("Potential spoofing detected")
                    continue
            
            filtered_encodings.append(encoding)
            filtered_locations.append(location)
        
        processing_time = time.time() - start_time
        logger.info(f"Processed {len(filtered_encodings)} faces in {processing_time:.2f}s")
        
        return filtered_encodings, filtered_locations
        
    except Exception as e:
        logger.error(f"Error processing image array: {e}")
        return [], []

def compare_faces_batch(known_encodings, unknown_encodings, tolerance=None):
    """Compare multiple unknown faces with known faces efficiently."""
    if tolerance is None:
        tolerance = FACE_RECOGNITION_CONFIG['tolerance']
    
    results = []
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=PERFORMANCE_CONFIG['max_workers']) as executor:
        futures = []
        
        for unknown_encoding in unknown_encodings:
            future = executor.submit(compare_faces, known_encodings, unknown_encoding, tolerance)
            futures.append(future)
        
        for future in futures:
            results.append(future.result())
    
    return results

def compare_faces(known_encodings, unknown_encoding, tolerance=None):
    """Compare unknown face with known faces with optimizations."""
    if unknown_encoding is None:
        return False, 0.0, -1
    
    if tolerance is None:
        tolerance = FACE_RECOGNITION_CONFIG['tolerance']
    
    # Compare faces
    matches = face_recognition.compare_faces(known_encodings, unknown_encoding, tolerance=tolerance)
    face_distances = face_recognition.face_distance(known_encodings, unknown_encoding)
    
    if len(face_distances) > 0:
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            confidence = 1 - face_distances[best_match_index]
            return True, confidence, best_match_index
    
    return False, 0.0, -1

def serialize_encoding(encoding):
    """Convert face encoding to bytes for database storage."""
    if encoding is None:
        return None
    return pickle.dumps(encoding.astype(np.float32))  # Use float32 to save space

def deserialize_encoding(encoded_bytes):
    """Convert bytes back to face encoding."""
    if encoded_bytes is None:
        return None
    return pickle.loads(encoded_bytes)

def detect_faces_in_frame(frame, enable_enhancement=True):
    """Detect all faces in a video frame with performance optimizations."""
    try:
        start_time = time.time()
        
        # Check system resources
        resources = monitor_system_resources()
        if resources['cpu_percent'] > 90 or resources['memory_percent'] > 90:
            logger.warning("High system resource usage, reducing processing quality")
            resize_factor = 0.5
        else:
            resize_factor = FACE_RECOGNITION_CONFIG['resize_factor']
        
        # Resize frame for faster processing
        height, width = frame.shape[:2]
        small_frame = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)
        
        # Convert BGR to RGB
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Enhance image quality if enabled
        if enable_enhancement:
            rgb_small_frame = enhance_image_quality(rgb_small_frame)
        
        # Find faces
        face_locations = face_recognition.face_locations(
            rgb_small_frame,
            model=FACE_RECOGNITION_CONFIG['model']
        )
        
        # Limit number of faces
        max_faces = FACE_RECOGNITION_CONFIG['max_faces_per_frame']
        if len(face_locations) > max_faces:
            face_locations = face_locations[:max_faces]
        
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, 
            face_locations,
            num_jitters=FACE_RECOGNITION_CONFIG['num_jitters']
        )
        
        # Scale back up face locations
        scale_factor = 1 / resize_factor
        face_locations = [(int(top * scale_factor), int(right * scale_factor), 
                          int(bottom * scale_factor), int(left * scale_factor)) 
                         for (top, right, bottom, left) in face_locations]
        
        processing_time = time.time() - start_time
        logger.info(f"Detected {len(face_encodings)} faces in {processing_time:.2f}s")
        
        return face_encodings, face_locations
        
    except Exception as e:
        logger.error(f"Error detecting faces in frame: {e}")
        return [], []

def draw_face_boxes(frame, face_locations, names, confidences, show_confidence=True):
    """Draw bounding boxes and names on detected faces with enhanced styling."""
    for (top, right, bottom, left), name, confidence in zip(face_locations, names, confidences):
        # Choose color based on recognition status
        if name != "Unknown":
            color = (0, 255, 0)  # Green for recognized
            if confidence < 0.7:
                color = (0, 255, 255)  # Yellow for low confidence
        else:
            color = (0, 0, 255)  # Red for unknown
        
        # Draw rectangle around face
        thickness = 2
        cv2.rectangle(frame, (left, top), (right, bottom), color, thickness)
        
        # Prepare label
        if name != "Unknown" and show_confidence:
            label = f"{name} ({confidence:.2f})"
        else:
            label = name
        
        # Calculate text size
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.6
        text_thickness = 1
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, text_thickness)
        
        # Draw label background
        cv2.rectangle(frame, (left, bottom - text_height - 10), 
                     (left + text_width, bottom), color, cv2.FILLED)
        
        # Draw label text
        cv2.putText(frame, label, (left + 6, bottom - 6), 
                   font, font_scale, (255, 255, 255), text_thickness)
    
    return frame

def get_optimal_face_encoding_settings():
    """Get optimal settings based on current system performance."""
    resources = monitor_system_resources()
    
    settings = {
        'model': 'hog',  # Default to faster model
        'num_jitters': 1,
        'resize_factor': 0.25,
        'max_faces': 70
    }
    
    # Adjust based on available resources
    if resources['memory_available_mb'] > 2000 and resources['cpu_percent'] < 50:
        settings['num_jitters'] = 2  # Higher accuracy
        settings['resize_factor'] = 0.3
    elif resources['cpu_percent'] > 80 or resources['memory_available_mb'] < 1000:
        settings['resize_factor'] = 0.2  # Faster processing
        settings['max_faces'] = 50
    
    return settings

def batch_process_student_photos(photo_paths, max_workers=None):
    """Process multiple student photos in parallel."""
    if max_workers is None:
        max_workers = min(PERFORMANCE_CONFIG['max_workers'], len(photo_paths))
    
    results = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {
            executor.submit(extract_face_encoding, path): path 
            for path in photo_paths
        }
        
        for future in future_to_path:
            path = future_to_path[future]
            try:
                result = future.result()
                results[path] = result
            except Exception as e:
                logger.error(f"Error processing {path}: {e}")
                results[path] = None
    
    return results

def clear_face_cache():
    """Clear the face recognition cache."""
    face_cache.clear()
    logger.info("Face recognition cache cleared")

def get_cache_stats():
    """Get cache statistics."""
    return {
        'cache_size': len(face_cache.cache),
        'max_size': face_cache.max_size,
        'hit_ratio': len(face_cache.cache) / face_cache.max_size if face_cache.max_size > 0 else 0
    }

def initialize_face_recognition_system():
    """Initialize and optimize the face recognition system."""
    logger.info("Initializing Face Recognition System...")
    
    # Check GPU availability and configure
    gpu_available, device_name = check_gpu_availability()
    
    # Update configuration based on hardware
    global FACE_RECOGNITION_CONFIG
    if gpu_available:
        FACE_RECOGNITION_CONFIG['model'] = 'cnn'  # Use CNN for GPU
        FACE_RECOGNITION_CONFIG['use_gpu'] = True
        logger.info(f"GPU mode enabled: {device_name}")
    else:
        FACE_RECOGNITION_CONFIG['model'] = 'hog'  # Use HOG for CPU
        FACE_RECOGNITION_CONFIG['use_gpu'] = False
        logger.info("CPU mode enabled")
    
    # Monitor initial system resources
    resources = monitor_system_resources()
    logger.info(f"System Resources - CPU: {resources['cpu_percent']:.1f}%, "
                f"Memory: {resources['memory_percent']:.1f}%, "
                f"Available: {resources['memory_available_mb']:.0f}MB")
    
    # Optimize settings based on available resources
    if resources['memory_available_mb'] < 2000:  # Less than 2GB available
        FACE_RECOGNITION_CONFIG['max_faces_per_frame'] = 50
        FACE_RECOGNITION_CONFIG['encoding_cache_size'] = 500
        FACE_RECOGNITION_CONFIG['resize_factor'] = 0.2
        logger.warning("Low memory detected - reducing processing limits")
    
    if resources['cpu_percent'] > 80:  # High CPU usage
        FACE_RECOGNITION_CONFIG['num_jitters'] = 1
        FACE_RECOGNITION_CONFIG['face_detection_batch_size'] = 4
        logger.warning("High CPU usage detected - reducing processing quality")
    
    logger.info("Face Recognition System initialized successfully")
    return True

def validate_system_requirements():
    """Validate all system requirements and dependencies."""
    issues = []
    
    try:
        # Check OpenCV
        cv2_version = cv2.__version__
        logger.info(f"OpenCV version: {cv2_version}")
    except Exception as e:
        issues.append(f"OpenCV issue: {e}")
    
    try:
        # Check face_recognition
        import face_recognition
        logger.info("face_recognition library loaded successfully")
    except Exception as e:
        issues.append(f"face_recognition issue: {e}")
    
    try:
        # Check torch for GPU
        if torch.cuda.is_available():
            torch_version = torch.__version__
            cuda_version = torch.version.cuda
            logger.info(f"PyTorch version: {torch_version}, CUDA: {cuda_version}")
        else:
            logger.info("PyTorch available but no CUDA support")
    except Exception as e:
        issues.append(f"PyTorch issue: {e}")
    
    try:
        # Check numpy version compatibility
        numpy_version = np.__version__
        logger.info(f"NumPy version: {numpy_version}")
    except Exception as e:
        issues.append(f"NumPy issue: {e}")
    
    # Check system resources
    resources = monitor_system_resources()
    if resources['memory_available_mb'] < 500:
        issues.append("Very low memory available (<500MB)")
    
    if resources['cpu_percent'] > 95:
        issues.append("Very high CPU usage (>95%)")
    
    # Check directories
    required_dirs = ['data/photos', 'data/temp', 'logs']
    for directory in required_dirs:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
                logger.info(f"Created directory: {directory}")
            except Exception as e:
                issues.append(f"Cannot create directory {directory}: {e}")
    
    if issues:
        logger.warning(f"System validation found {len(issues)} issues:")
        for issue in issues:
            logger.warning(f"  - {issue}")
        return False, issues
    else:
        logger.info("System validation passed - all requirements met")
        return True, []

def sanitize_numpy_types(data):
    """Convert NumPy types to Python native types for database compatibility."""
    if isinstance(data, dict):
        return {k: sanitize_numpy_types(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_numpy_types(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(sanitize_numpy_types(item) for item in data)
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.bool_):
        return bool(data)
    else:
        return data

def safe_extract_face_encoding(image_path, enhance_quality=True):
    """Safe wrapper for extract_face_encoding with proper error handling and type conversion."""
    try:
        result = extract_face_encoding(image_path, enhance_quality)
        
        if result is None:
            return None
        
        if isinstance(result, tuple) and len(result) == 2:
            encoding, quality_score = result
            # Ensure quality_score is a Python float
            quality_score = float(quality_score) if quality_score is not None else 0.0
            return encoding, float(quality_score)
        else:
            logger.warning(f"Unexpected result format from extract_face_encoding: {type(result)}")
            return None
            
    except Exception as e:
        logger.error(f"Error in safe_extract_face_encoding for {image_path}: {e}")
        return None

def batch_validate_face_data(face_data_list):
    """Validate and sanitize a batch of face data before database insertion."""
    validated_data = []
    
    for i, face_data in enumerate(face_data_list):
        try:
            # Sanitize all numpy types
            sanitized_data = sanitize_numpy_types(face_data)
            
            # Validate required fields
            if 'quality_score' in sanitized_data:
                quality_score = sanitized_data['quality_score']
                if not isinstance(quality_score, (int, float)):
                    logger.warning(f"Invalid quality_score type in item {i}: {type(quality_score)}")
                    sanitized_data['quality_score'] = 0.0
                else:
                    sanitized_data['quality_score'] = float(quality_score)
            
            validated_data.append(sanitized_data)
            
        except Exception as e:
            logger.error(f"Error validating face data item {i}: {e}")
            continue
    
    return validated_data

def get_system_performance_report():
    """Generate a comprehensive system performance report."""
    report = {
        'timestamp': time.time(),
        'hardware': {},
        'memory': {},
        'cache': {},
        'performance': {}
    }
    
    try:
        # GPU information
        gpu_available, device_name = check_gpu_availability()
        report['hardware']['gpu_available'] = gpu_available
        report['hardware']['gpu_device'] = device_name
        
        # System resources
        resources = monitor_system_resources()
        report['hardware'].update(resources)
        
        # Cache statistics
        cache_stats = get_cache_stats()
        report['cache'] = cache_stats
        
        # Memory details
        memory = psutil.virtual_memory()
        report['memory'] = {
            'total_mb': memory.total / (1024 * 1024),
            'available_mb': memory.available / (1024 * 1024),
            'used_mb': memory.used / (1024 * 1024),
            'percent': memory.percent
        }
        
        # Performance configuration
        report['performance'] = {
            'model': FACE_RECOGNITION_CONFIG['model'],
            'max_faces': FACE_RECOGNITION_CONFIG['max_faces_per_frame'],
            'cache_size': FACE_RECOGNITION_CONFIG['encoding_cache_size'],
            'use_gpu': FACE_RECOGNITION_CONFIG['use_gpu']
        }
        
        logger.info("System performance report generated successfully")
        
    except Exception as e:
        logger.error(f"Error generating performance report: {e}")
        report['error'] = str(e)
    
    return report

# Initialize the system when module is imported
try:
    initialize_face_recognition_system()
    is_valid, validation_issues = validate_system_requirements()
    if not is_valid:
        logger.warning("System validation failed - some features may not work properly")
except Exception as e:
    logger.error(f"Error during face recognition system initialization: {e}") 