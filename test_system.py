#!/usr/bin/env python3
"""
System Test Script for Multi-Face Attendance System
Tests GPU availability, system performance, and face recognition capabilities
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.face_utils import (
    check_gpu_availability, validate_system_requirements, 
    get_system_performance_report, initialize_face_recognition_system
)
import torch
import cv2
import face_recognition
import numpy as np
import time

def test_gpu_support():
    """Test GPU support and capabilities."""
    print("=" * 60)
    print("GPU SUPPORT TEST")
    print("=" * 60)
    
    # PyTorch CUDA test
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: {gpu_name} ({memory:.1f} GB)")
    
    # Face recognition GPU check
    gpu_available, device_name = check_gpu_availability()
    print(f"Face Recognition GPU support: {gpu_available}")
    print(f"Device: {device_name}")
    
    return gpu_available

def test_face_recognition_performance():
    """Test face recognition performance."""
    print("\n" + "=" * 60)
    print("FACE RECOGNITION PERFORMANCE TEST")
    print("=" * 60)
    
    # Create a test image
    test_image = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
    
    # Test face detection speed
    print("Testing face detection speed...")
    start_time = time.time()
    
    try:
        face_locations = face_recognition.face_locations(test_image, model='hog')
        hog_time = time.time() - start_time
        print(f"HOG model time: {hog_time:.4f} seconds")
        
        start_time = time.time()
        face_locations = face_recognition.face_locations(test_image, model='cnn')
        cnn_time = time.time() - start_time
        print(f"CNN model time: {cnn_time:.4f} seconds")
        
        if cnn_time < hog_time:
            print("✅ CNN model is faster - GPU acceleration likely working")
        else:
            print("⚠️ HOG model is faster - using CPU processing")
            
    except Exception as e:
        print(f"❌ Error testing face recognition: {e}")

def test_system_resources():
    """Test system resource availability."""
    print("\n" + "=" * 60)
    print("SYSTEM RESOURCES TEST")
    print("=" * 60)
    
    # Get performance report
    report = get_system_performance_report()
    
    print(f"CPU Usage: {report['hardware']['cpu_percent']:.1f}%")
    print(f"Memory Usage: {report['hardware']['memory_percent']:.1f}%")
    print(f"Available Memory: {report['hardware']['memory_available_mb']:.0f} MB")
    print(f"Total Memory: {report['memory']['total_mb']:.0f} MB")
    
    # Check if system can handle 70 faces
    available_memory_gb = report['memory']['available_mb'] / 1024
    print(f"Available Memory: {available_memory_gb:.1f} GB")
    
    if available_memory_gb >= 2.0:
        print("✅ Sufficient memory for 70-face processing")
    else:
        print("⚠️ Limited memory - may need to reduce face count")

def test_opencv_features():
    """Test OpenCV features and optimizations."""
    print("\n" + "=" * 60)
    print("OPENCV FEATURES TEST")
    print("=" * 60)
    
    print(f"OpenCV version: {cv2.__version__}")
    
    # Check for optimizations
    print("\nOptimizations available:")
    print(f"OpenMP: {'Yes' if cv2.getBuildInformation().find('OpenMP') != -1 else 'No'}")
    print(f"Intel IPP: {'Yes' if cv2.getBuildInformation().find('IPP') != -1 else 'No'}")
    print(f"Intel TBB: {'Yes' if cv2.getBuildInformation().find('TBB') != -1 else 'No'}")
    
    # Test basic operations
    test_img = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
    
    start_time = time.time()
    resized = cv2.resize(test_img, (500, 500))
    resize_time = time.time() - start_time
    print(f"Image resize time: {resize_time:.4f} seconds")
    
    start_time = time.time()
    gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    convert_time = time.time() - start_time
    print(f"Color conversion time: {convert_time:.4f} seconds")

def main():
    """Run all system tests."""
    print("🔧 MULTI-FACE ATTENDANCE SYSTEM - SYSTEM TEST")
    print("Optimized for Lenovo Legion 5i 2022 i7 12th Gen")
    print("Testing GPU support, performance, and capabilities...\n")
    
    # Initialize system
    try:
        initialize_face_recognition_system()
        print("✅ Face recognition system initialized successfully\n")
    except Exception as e:
        print(f"❌ Error initializing system: {e}\n")
    
    # Validate requirements
    is_valid, issues = validate_system_requirements()
    if is_valid:
        print("✅ All system requirements validated\n")
    else:
        print("⚠️ System validation issues found:")
        for issue in issues:
            print(f"  - {issue}")
        print()
    
    # Run tests
    gpu_available = test_gpu_support()
    test_face_recognition_performance()
    test_system_resources()
    test_opencv_features()
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    if gpu_available:
        print("✅ GPU acceleration available")
        print("✅ CNN model recommended for better accuracy")
    else:
        print("⚠️ GPU acceleration not available")
        print("⚠️ Using CPU processing with HOG model")
    
    print("\n🚀 System ready for multi-face attendance processing!")
    print("📱 Access the web app at: http://localhost:8501")

if __name__ == "__main__":
    main() 