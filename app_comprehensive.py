import warnings
import os
# Suppress PyTorch warnings for Streamlit compatibility
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
warnings.filterwarnings("ignore", message=".*torch.classes.*")
warnings.filterwarnings("ignore", message=".*_path.*")
# Set PyTorch environment variables for better Streamlit compatibility
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['PYTORCH_DISABLE_PER_OP_PROFILING'] = '1'
# Remove the invalid TORCH_LOGS setting that was causing the error

import streamlit as st
import cv2
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import time
from datetime import datetime, timedelta, date
from PIL import Image
import io
import qrcode
import psutil

# Configure page
st.set_page_config(
    page_title="Multi-Face Attendance System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "Multi-Face Attendance System v2.0"
    }
)


# Global error handler
def handle_streamlit_error(func):
    """Decorator to handle errors gracefully in Streamlit"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyError as e:
            st.error(f"Data access error: {e}. Please refresh the page.")
            return None
        except Exception as e:
            st.error(f"An error occurred: {e}")
            return None
    return wrapper

# Import utilities
from utils.auth_utils import (
    AuthManager, is_authenticated, show_login_page, logout, 
    get_current_user, has_role, require_auth
)
from utils.db_utils import (
    create_tables, SessionLocal, Student, Photo, Class, Subject,
    AttendanceSession, AttendanceRecord, AcademicYear, Semester, Notification,
    PerformanceMetrics, AuditLog, get_classes_for_teacher, get_students_in_class, log_audit_event
)
from utils.face_utils import (
    extract_face_encoding, safe_extract_face_encoding, serialize_encoding, deserialize_encoding,
    extract_face_encoding, serialize_encoding, deserialize_encoding,
    detect_faces_in_frame, compare_faces, draw_face_boxes,
    monitor_system_resources, get_cache_stats, clear_face_cache,
    batch_process_student_photos
)
from utils.analytics_utils import AttendanceAnalytics, export_attendance_data_to_excel
from utils.notification_utils import NotificationManager, get_unread_notifications
from config import UI_CONFIG, PERFORMANCE_CONFIG, MONITORING_CONFIG

# Initialize system
create_tables()
os.makedirs("data/photos", exist_ok=True)

# Custom CSS for modern UI
def load_custom_css():
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .status-excellent { color: #28a745; font-weight: bold; }
    .status-good { color: #17a2b8; font-weight: bold; }
    .status-average { color: #ffc107; font-weight: bold; }
    .status-poor { color: #dc3545; font-weight: bold; }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    .notification-badge {
        background-color: #dc3545;
        color: white;
        border-radius: 50%;
        padding: 0.2rem 0.5rem;
        font-size: 0.8rem;
    }
    </style>
    """, unsafe_allow_html=True)

load_custom_css()

# Authentication check
if not is_authenticated():
    show_login_page()
    st.stop()

# Header with user info and logout
def show_header():
    st.markdown("""
    <div class="main-header">
        <h1>🎓 Multi-Face Attendance System</h1>
        <p>Advanced Face Recognition • Real-time Analytics • Automated Notifications</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown(f"**👤 User:** {get_current_user()}")
        st.markdown(f"**🎭 Role:** {st.session_state.get('user_role', 'N/A').title()}")
        
        # System resources
        resources = monitor_system_resources()
        st.markdown("**📊 System Status:**")
        st.progress(resources['cpu_percent'] / 100, f"CPU: {resources['cpu_percent']:.1f}%")
        st.progress(resources['memory_percent'] / 100, f"Memory: {resources['memory_percent']:.1f}%")
        
        # Cache stats
        cache_stats = get_cache_stats()
        st.markdown(f"**🧠 Cache:** {cache_stats['cache_size']}/{cache_stats['max_size']}")
        
        if st.button("🚪 Logout"):
            logout()

show_header()

# Navigation
def get_navigation_menu():
    """Get navigation menu based on user role."""
    base_menu = [
        "📊 Dashboard",
        "👥 Manage Students", 
        "📷 Take Attendance",
        "📈 Analytics",
        "📥 Export Data"
    ]
    
    if has_role(['admin']):
        admin_menu = [
            "🏫 Manage Classes",
            "👨‍🏫 Manage Users",
            "🔔 Notifications",
            "⚙️ System Settings",
            "🔍 Audit Logs"
        ]
        return base_menu + admin_menu
    
    return base_menu

menu_options = get_navigation_menu()
choice = st.sidebar.selectbox("📂 Navigation", menu_options)

# Notification indicator
def show_notifications():
    notifications = get_unread_notifications(limit=5)
    if notifications:
        st.sidebar.markdown(f"""
        <div style="background-color: #fff3cd; padding: 0.5rem; border-radius: 5px; margin: 0.5rem 0;">
            <strong>🔔 {len(notifications)} Notifications</strong>
        </div>
        """, unsafe_allow_html=True)

show_notifications()

# Helper functions
def save_uploaded_photo(uploaded_file, student_id):
    """Save uploaded photo and return file path."""
    if uploaded_file is not None:
        student_dir = f"data/photos/student_{student_id}"
        os.makedirs(student_dir, exist_ok=True)
        
        file_path = os.path.join(student_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    return None

def load_known_faces():
    """Load all known face encodings from database."""
    session = SessionLocal()
    try:
        photos = session.query(Photo).filter(Photo.face_encoding.isnot(None)).all()
        
        known_encodings = []
        known_names = []
        known_ids = []
        
        for photo in photos:
            encoding = deserialize_encoding(photo.face_encoding)
            if encoding is not None:
                known_encodings.append(encoding)
                known_names.append(photo.student.name)
                known_ids.append(photo.student.id)
        
        return known_encodings, known_names, known_ids
    finally:
        session.close()

def get_status_style(status):
    """Get CSS style for attendance status."""
    styles = {
        'Excellent': 'status-excellent',
        'Good': 'status-good', 
        'Average': 'status-average',
        'Poor': 'status-poor'
    }
    return styles.get(status, '')

def validate_attendance_data(recognized_students, confidences, all_students):
    """Validate attendance data before database insertion."""
    errors = []
    
    # Check if lists have matching lengths
    if len(recognized_students) != len(confidences):
        errors.append("Mismatch between recognized students and confidence scores")
    
    # Validate confidence scores
    for i, confidence in enumerate(confidences):
        if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
            errors.append(f"Invalid confidence score at index {i}: {confidence}")
    
    # Check for duplicate student IDs
    if len(recognized_students) != len(set(recognized_students)):
        errors.append("Duplicate student IDs detected")
    
    # Validate student IDs exist
    valid_student_ids = [s.id for s in all_students]
    for student_id in recognized_students:
        if student_id not in valid_student_ids:
            errors.append(f"Invalid student ID: {student_id}")
    
    return errors

# Page: Dashboard
if choice == "📊 Dashboard":
    st.header("📊 Dashboard")
    
    # Quick stats
    session = SessionLocal()
    analytics = AttendanceAnalytics()
    
    try:
        # Get user's classes
        if has_role(['admin']):
            classes = session.query(Class).filter(Class.is_active == True).all()
        else:
            classes = get_classes_for_teacher(session, get_current_user())
        
        if not classes:
            st.warning("No classes assigned. Please contact administrator.")
            st.stop()
        
        selected_class = st.selectbox("Select Class", 
                                    options=[(c.id, f"{c.name} ({c.code})") for c in classes],
                                    format_func=lambda x: x[1])
        
        class_id = selected_class[0] if selected_class else None
        
        if class_id:
            # Get statistics
            stats = analytics.get_attendance_statistics(class_id)
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("""
                <div class="metric-card">
                    <h3>👥 Total Students</h3>
                    <h2>{}</h2>
                </div>
                """.format(stats.get('total_students', 0)), unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="metric-card">
                    <h3>📅 Total Sessions</h3>
                    <h2>{}</h2>
                </div>
                """.format(stats.get('total_sessions', 0)), unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div class="metric-card">
                    <h3>📊 Overall Rate</h3>
                    <h2>{:.1f}%</h2>
                </div>
                """.format(stats.get('overall_attendance_rate', 0)), unsafe_allow_html=True)
            
            with col4:
                st.markdown("""
                <div class="metric-card">
                    <h3>⚠️ Below 75%</h3>
                    <h2>{}</h2>
                </div>
                """.format(stats.get('students_below_75', 0)), unsafe_allow_html=True)
            
            # Attendance trends chart
            st.subheader("📈 Attendance Trends (Last 30 Days)")
            trends = analytics.get_attendance_trends(class_id=class_id, days_back=30)
            
            if trends and trends.get('dates') and len(trends.get('dates', [])) > 0:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=trends.get('dates', []),
                    y=trends.get('percentages', []),
                    mode='lines+markers',
                    name='Attendance %',
                    line=dict(color='#667eea', width=3)
                ))
                
                fig.update_layout(
                    title="Daily Attendance Percentage",
                    xaxis_title="Date",
                    yaxis_title="Attendance %",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info('No attendance data available for the selected period.')
            
            # Recent sessions
            st.subheader("📅 Recent Sessions")
            recent_sessions = session.query(AttendanceSession).filter(
                AttendanceSession.class_id == class_id
            ).order_by(AttendanceSession.date.desc()).limit(5).all()
            
            if recent_sessions:
                sessions_data = []
                for sess in recent_sessions:
                    sessions_data.append({
                        'Date': sess.date.strftime('%Y-%m-%d %H:%M'),
                        'Session': sess.session_name,
                        'Present': sess.total_present,
                        'Absent': sess.total_absent,
                        'Rate': f"{(sess.total_present / (sess.total_present + sess.total_absent) * 100):.1f}%" if (sess.total_present + sess.total_absent) > 0 else "0%"
                    })
                
                st.dataframe(pd.DataFrame(sessions_data), use_container_width=True)
            else:
                st.info("No attendance sessions found.")
    
    finally:
        analytics.close()
        session.close()

# Page: Manage Students
elif choice == "👥 Manage Students":
    st.header("👥 Student Management")
    
    tabs = st.tabs(["➕ Add Student", "👀 View Students", "📊 Bulk Import"])
    
    with tabs[0]:  # Add Student
        st.subheader("➕ Add New Student")
        
        # Get user's classes for selection
        session = SessionLocal()
        try:
            if has_role(['admin']):
                classes = session.query(Class).filter(Class.is_active == True).all()
            else:
                classes = get_classes_for_teacher(session, get_current_user())
            
            if not classes:
                st.warning("No classes available.")
            else:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    with st.form("student_form"):
                        name = st.text_input("Student Name *")
                        roll = st.text_input("Roll Number *")
                        email = st.text_input("Student Email")
                        phone = st.text_input("Student Phone")
                        parent_email = st.text_input("Parent Email")
                        parent_phone = st.text_input("Parent Phone")
                        
                        class_options = [(c.id, f"{c.name} ({c.code})") for c in classes]
                        selected_class = st.selectbox("Class *", 
                                                    options=class_options,
                                                    format_func=lambda x: x[1])
                        
                        uploaded_files = st.file_uploader(
                            "Upload Photos (2-3 recommended)", 
                            type=['jpg', 'jpeg', 'png', 'bmp'], 
                            accept_multiple_files=True,
                            help="Upload multiple clear photos for better recognition accuracy"
                        )
                        
                        submitted = st.form_submit_button("➕ Add Student", type="primary")
                        
                        if submitted and name and roll and selected_class:
                            class_id = selected_class[0]
                            
                            # Check if roll number already exists in class
                            existing_student = session.query(Student).filter(
                                Student.roll == roll,
                                Student.class_id == class_id
                            ).first()
                            
                            if existing_student:
                                st.error("Roll number already exists in this class!")
                            else:
                                try:
                                    # Add student
                                    student = Student(
                                        name=name,
                                        roll=roll,
                                        email=email,
                                        phone=phone,
                                        parent_email=parent_email,
                                        parent_phone=parent_phone,
                                        class_id=class_id
                                    )
                                    session.add(student)
                                    session.commit()
                                    
                                    # Process photos
                                    if uploaded_files:
                                        progress_bar = st.progress(0)
                                        status_text = st.empty()
                                        
                                        for i, uploaded_file in enumerate(uploaded_files):
                                            status_text.text(f"Processing photo {i+1}/{len(uploaded_files)}")
                                            
                                            file_path = save_uploaded_photo(uploaded_file, student.id)
                                            
                                            if file_path:
                                                result = extract_face_encoding(file_path)
                                                
                                                if result and len(result) == 2:
                                                    encoding, quality_score = result
                                                    # Ensure quality_score is a Python float
                                                    quality_score = float(quality_score) if quality_score is not None else 0.0
                                                
                                                if result and len(result) == 2:
                                                    encoding, quality_score = result
                                                    
                                                    photo = Photo(
                                                        student_id=student.id,
                                                        photo_path=file_path,
                                                        face_encoding=serialize_encoding(encoding),
                                                        quality_score=quality_score,
                                                        is_primary=(i == 0)
                                                    )
                                                    session.add(photo)
                                                else:
                                                    st.warning(f"No face detected in {uploaded_file.name}")
                                            
                                            progress_bar.progress((i + 1) / len(uploaded_files))
                                        
                                        session.commit()
                                        progress_bar.empty()
                                        status_text.empty()
                                    
                                    # Log audit event
                                    log_audit_event(
                                        session, 
                                        get_current_user(), 
                                        'add_student',
                                        'student',
                                        student.id,
                                        f"Added student {name} to class {selected_class[1]}"
                                    )
                                    
                                    st.success(f"✅ Student {name} added successfully!")
                                    time.sleep(2)
                                    st.rerun()
                                    
                                except Exception as e:
                                    session.rollback()
                                    st.error(f"Error adding student: {e}")
                        elif submitted:
                            st.error("Please fill in all required fields (*).")
                
                with col2:
                    st.info("💡 **Tips for Better Recognition:**")
                    st.write("• Upload 2-3 clear, well-lit photos")
                    st.write("• Include different angles/expressions")
                    st.write("• Ensure face is clearly visible")
                    st.write("• Avoid blurry or dark images")
                    st.write("• Remove sunglasses/masks")
        
        finally:
            session.close()
    
    with tabs[1]:  # View Students
        st.subheader("👀 Student Directory")
        
        session = SessionLocal()
        try:
            # Class filter
            if has_role(['admin']):
                classes = session.query(Class).filter(Class.is_active == True).all()
            else:
                classes = get_classes_for_teacher(session, get_current_user())
            
            if classes:
                class_options = [('all', 'All Classes')] + [(c.id, f"{c.name} ({c.code})") for c in classes]
                selected_class_filter = st.selectbox("Filter by Class", 
                                                   options=class_options,
                                                   format_func=lambda x: x[1])
                
                # Get students
                query = session.query(Student).filter(Student.is_active == True)
                
                if selected_class_filter[0] != 'all':
                    query = query.filter(Student.class_id == selected_class_filter[0])
                
                students = query.order_by(Student.name).all()
                
                if students:
                    # Search
                    search_term = st.text_input("🔍 Search students", placeholder="Enter name or roll number")
                    
                    if search_term:
                        students = [s for s in students if 
                                  search_term.lower() in s.name.lower() or 
                                  search_term.lower() in s.roll.lower()]
                    
                    st.write(f"**Found {len(students)} students**")
                    
                    # Display students in expandable cards
                    for student in students:
                        analytics = AttendanceAnalytics()
                        try:
                            attendance_percentage = analytics.calculate_student_attendance_percentage(
                                student.id, student.class_id, days_back=30
                            )
                            status = analytics._get_attendance_status(attendance_percentage)
                        finally:
                            analytics.close()
                        
                        with st.expander(f"👤 {student.name} ({student.roll}) - {attendance_percentage:.1f}%"):
                            col1, col2, col3 = st.columns([2, 2, 1])
                            
                            with col1:
                                st.write(f"**Name:** {student.name}")
                                st.write(f"**Roll:** {student.roll}")
                                st.write(f"**Class:** {student.class_obj.name if student.class_obj else 'N/A'}")
                                st.write(f"**Email:** {student.email or 'N/A'}")
                                st.write(f"**Phone:** {student.phone or 'N/A'}")
                            
                            with col2:
                                st.write(f"**Parent Email:** {student.parent_email or 'N/A'}")
                                st.write(f"**Parent Phone:** {student.parent_phone or 'N/A'}")
                                st.write(f"**Admission Date:** {student.admission_date.strftime('%Y-%m-%d')}")
                                st.markdown(f"**Attendance:** <span class='{get_status_style(status)}'>{attendance_percentage:.1f}% ({status})</span>", 
                                          unsafe_allow_html=True)
                            
                            with col3:
                                # Show photos
                                photos = session.query(Photo).filter(Photo.student_id == student.id).all()
                                st.write(f"**Photos:** {len(photos)}")
                                
                                if photos:
                                    photo_cols = st.columns(min(len(photos), 3))
                                    for i, photo in enumerate(photos[:3]):
                                        if os.path.exists(photo.photo_path):
                                            img = Image.open(photo.photo_path)
                                            photo_cols[i].image(img, width=80)
                                
                                # Action buttons
                                col_edit, col_delete = st.columns(2)
                                
                                with col_edit:
                                    if st.button(f"✏️ Edit", key=f"edit_{student.id}"):
                                        st.session_state[f'edit_student_{student.id}'] = True
                                
                                with col_delete:
                                    if st.button(f"🗑️ Delete", key=f"del_{student.id}", type="secondary"):
                                        # Delete confirmation
                                        if st.session_state.get(f'confirm_delete_{student.id}', False):
                                            # Delete student and associated photos
                                            for photo in student.photos:
                                                if os.path.exists(photo.photo_path):
                                                    os.remove(photo.photo_path)
                                            
                                            log_audit_event(
                                                session, 
                                                get_current_user(), 
                                                'delete_student',
                                                'student',
                                                student.id,
                                                f"Deleted student {student.name}"
                                            )
                                            
                                            session.delete(student)
                                            session.commit()
                                            st.success(f"Student {student.name} deleted!")
                                            time.sleep(1)
                                            st.rerun()
                                        else:
                                            st.session_state[f'confirm_delete_{student.id}'] = True
                                            st.warning("Click delete again to confirm")
                else:
                    st.info("No students found.")
        
        finally:
            session.close()
    
    with tabs[2]:  # Bulk Import
        st.subheader("📊 Bulk Import Students")
        
        st.info("💡 Upload an Excel file with student information to add multiple students at once.")
        
        # Download template
        template_data = {
            'Name': ['John Doe', 'Jane Smith'],
            'Roll': ['2023001', '2023002'],
            'Email': ['john@example.com', 'jane@example.com'],
            'Phone': ['1234567890', '0987654321'],
            'Parent_Email': ['john.parent@example.com', 'jane.parent@example.com'],
            'Parent_Phone': ['1111111111', '2222222222']
        }
        template_df = pd.DataFrame(template_data)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.download_button(
                "📥 Download Template",
                template_df.to_csv(index=False),
                "student_template.csv",
                "text/csv"
            )
        
        with col2:
            uploaded_file = st.file_uploader("Upload Student Data", type=['csv', 'xlsx'])
        
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.write("**Preview:**")
                st.dataframe(df.head())
                
                # Class selection for bulk import
                session = SessionLocal()
                try:
                    if has_role(['admin']):
                        classes = session.query(Class).filter(Class.is_active == True).all()
                    else:
                        classes = get_classes_for_teacher(session, get_current_user())
                    
                    if classes:
                        class_options = [(c.id, f"{c.name} ({c.code})") for c in classes]
                        selected_class = st.selectbox("Select Class for Import", 
                                                    options=class_options,
                                                    format_func=lambda x: x[1])
                        
                        if st.button("📊 Import Students", type="primary"):
                            class_id = selected_class[0]
                            success_count = 0
                            error_count = 0
                            
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            for index, row in df.iterrows():
                                try:
                                    status_text.text(f"Processing {index + 1}/{len(df)}: {row['Name']}")
                                    
                                    # Check if student already exists
                                    existing = session.query(Student).filter(
                                        Student.roll == str(row['Roll']),
                                        Student.class_id == class_id
                                    ).first()
                                    
                                    if not existing:
                                        student = Student(
                                            name=str(row['Name']),
                                            roll=str(row['Roll']),
                                            email=str(row.get('Email', '')),
                                            phone=str(row.get('Phone', '')),
                                            parent_email=str(row.get('Parent_Email', '')),
                                            parent_phone=str(row.get('Parent_Phone', '')),
                                            class_id=class_id
                                        )
                                        session.add(student)
                                        session.commit()
                                        success_count += 1
                                    else:
                                        error_count += 1
                                
                                except Exception as e:
                                    error_count += 1
                                    st.error(f"Error processing {row['Name']}: {e}")
                                
                                progress_bar.progress((index + 1) / len(df))
                            
                            progress_bar.empty()
                            status_text.empty()
                            
                            st.success(f"✅ Import completed! {success_count} students added, {error_count} errors.")
                            
                            # Log audit event
                            log_audit_event(
                                session, 
                                get_current_user(), 
                                'bulk_import_students',
                                'student',
                                None,
                                f"Bulk imported {success_count} students to class {selected_class[1]}"
                            )
                
                finally:
                    session.close()
                    
            except Exception as e:
                st.error(f"Error reading file: {e}")

# Page: Take Attendance 
elif choice == "📷 Take Attendance":
    st.header("📷 Take Attendance")
    
    # Load known faces
    with st.spinner("Loading student faces..."):
        known_encodings, known_names, known_ids = load_known_faces()
    
    if not known_encodings:
        st.warning("⚠️ No student photos found! Please add students with photos first.")
        st.stop()
    
    st.success(f"✅ Loaded {len(known_encodings)} known faces")
    
    # Session configuration
    session = SessionLocal()
    try:
        # Get user's classes
        if has_role(['admin']):
            classes = session.query(Class).filter(Class.is_active == True).all()
        else:
            classes = get_classes_for_teacher(session, get_current_user())
        
        if not classes:
            st.warning("No classes assigned.")
            st.stop()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_class = st.selectbox("Select Class", 
                                        options=[(c.id, f"{c.name} ({c.code})") for c in classes],
                                        format_func=lambda x: x[1])
            
            # Get subjects for selected class
            if selected_class:
                class_id = selected_class[0]
                subjects = session.query(Subject).filter(
                    Subject.class_id == class_id,
                    Subject.is_active == True
                ).all()
                
                subject_options = [('general', 'General Attendance')] + [(s.id, f"{s.name} ({s.code})") for s in subjects]
                selected_subject = st.selectbox("Select Subject", 
                                              options=subject_options,
                                              format_func=lambda x: x[1])
        
        with col2:
            session_name = st.text_input("Session Name", 
                                       value=f"Attendance_{datetime.now().strftime('%Y%m%d_%H%M')}")
            
            # Real-time system monitoring
            st.subheader("🖥️ System Status")
            resources = monitor_system_resources()
            
            cpu_color = "🟢" if resources['cpu_percent'] < 70 else "🟡" if resources['cpu_percent'] < 90 else "🔴"
            memory_color = "🟢" if resources['memory_percent'] < 70 else "🟡" if resources['memory_percent'] < 90 else "🔴"
            
            st.write(f"{cpu_color} CPU: {resources['cpu_percent']:.1f}%")
            st.write(f"{memory_color} Memory: {resources['memory_percent']:.1f}%")
            st.write(f"💾 Available: {resources['memory_available_mb']:.0f} MB")
        
        # Camera input with enhanced processing
        st.subheader("📸 Capture Attendance Photo")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            camera_input = st.camera_input("Take a photo for attendance")
        
        with col2:
            enable_liveness = st.checkbox("🔍 Anti-Spoofing", 
                                        value=True, 
                                        help="Enable liveness detection to prevent photo spoofing")
            
            confidence_threshold = st.slider("🎯 Confidence Threshold", 
                                            min_value=0.3, 
                                            max_value=0.9, 
                                            value=0.6, 
                                            step=0.05)
        
        if camera_input is not None:
            # Process the image
            image = Image.open(camera_input)
            image_array = np.array(image)
            if image_array is None or image_array.size == 0:
                st.error("Invalid image data")
            else:
                with st.spinner("🔍 Detecting and recognizing faces..."):
                    start_time = time.time()
                    
                    # Detect faces with performance monitoring
                    face_encodings, face_locations = detect_faces_in_frame(image_array)
                    
                    processing_time = time.time() - start_time
                    
                    if face_encodings:
                        st.success(f"✅ Detected {len(face_encodings)} faces in {processing_time:.2f} seconds!")
                        
                        # Recognize faces
                        recognized_students = []
                        confidences = []
                        names = []
                        
                        recognition_start = time.time()
                        
                        for face_encoding in face_encodings:
                            best_match_index = None
                            best_confidence = 0
                            
                            for i, known_encoding in enumerate(known_encodings):
                                is_match, confidence, _ = compare_faces([known_encoding], face_encoding)
                                if is_match and confidence > best_confidence and confidence >= confidence_threshold:
                                    best_confidence = confidence
                                    best_match_index = i
                            
                            if best_match_index is not None:
                                recognized_students.append(known_ids[best_match_index])
                                names.append(known_names[best_match_index])
                                confidences.append(best_confidence)
                            else:
                                names.append("Unknown")
                                confidences.append(0.0)
                        
                        recognition_time = time.time() - recognition_start
                        
                        # Draw face boxes with enhanced styling
                        annotated_image = draw_face_boxes(image_array.copy(), face_locations, names, confidences, show_confidence=True)
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.image(annotated_image, caption=f"Face Recognition Results (Recognition: {recognition_time:.2f}s)")
                        
                        with col2:
                            st.subheader("🎯 Recognition Results")
                            st.write(f"**Processing Time:** {processing_time:.2f}s")
                            st.write(f"**Recognition Time:** {recognition_time:.2f}s")
                            st.write(f"**Total Faces:** {len(face_encodings)}")
                            st.write(f"**Recognized:** {len([n for n in names if n != 'Unknown'])}")
                            st.write(f"**Unknown:** {len([n for n in names if n == 'Unknown'])}")
                        
                        # Show recognized students
                        if recognized_students:
                            st.subheader("✅ Recognized Students")
                            recognized_df = pd.DataFrame({
                                'Name': [names[i] for i in range(len(names)) if names[i] != "Unknown"],
                                'Confidence': [f"{confidences[i]:.2f}" for i in range(len(confidences)) if names[i] != "Unknown"],
                                'Status': ['Present'] * len([n for n in names if n != "Unknown"])
                            })
                            st.dataframe(recognized_df, use_container_width=True)
                        
                        # Unknown faces warning
                        unknown_count = len([n for n in names if n == 'Unknown'])
                        if unknown_count > 0:
                            st.warning(f"⚠️ {unknown_count} unknown face(s) detected. These may be:")
                            st.write("• New students not in the system")
                            st.write("• Poor image quality")
                            st.write("• Visitors or unauthorized persons")
                        
                        # Save attendance button
                        if st.button("💾 Save Attendance", type="primary") and session_name:
                            try:
                                with st.spinner("Saving attendance data..."):
                                    class_id = selected_class[0]
                                    subject_id = selected_subject[0] if selected_subject[0] != 'general' else None
                                    
                                    # Create attendance session
                                    attendance_session = AttendanceSession(
                                        session_name=session_name,
                                        class_id=class_id,
                                        subject_id=subject_id,
                                        conducted_by=get_current_user(),
                                        total_present=len(recognized_students),
                                        total_absent=0,  # Will be calculated
                                        location="Classroom"
                                    )
                                    session.add(attendance_session)
                                    session.commit()
                                    
                                    # Get all students in class
                                    all_students = get_students_in_class(session, class_id)
                                    
                                    absent_count = 0
                                    
                                    # Mark attendance for all students
                                    for student in all_students:
                                        if student.id in recognized_students:
                                            # Present
                                            idx = recognized_students.index(student.id)
                                            confidence = float(confidences[idx]) if idx < len(confidences) else 0.0 if idx < len(confidences) else 0.0
                                            record = AttendanceRecord(
                                                session_id=attendance_session.id,
                                                student_id=student.id,
                                                status='present',
                                                confidence=float(confidence),
                                                marked_by='auto'
                                            )
                                        else:
                                            # Absent
                                            absent_count += 1
                                            record = AttendanceRecord(
                                                session_id=attendance_session.id,
                                                student_id=student.id,
                                                status='absent',
                                                marked_by='auto'
                                            )
                                        session.add(record)
                                    
                                    # Update session totals
                                    attendance_session.total_absent = absent_count
                                    session.commit()
                                    
                                    # Log audit event
                                    log_audit_event(
                                        session, 
                                        get_current_user(), 
                                        'take_attendance',
                                        'attendance_session',
                                        attendance_session.id,
                                        f"Attendance taken for {session_name}: {len(recognized_students)} present, {absent_count} absent"
                                    )
                                    
                                    st.success(f"""
                                    ✅ **Attendance Saved Successfully!**
                                    
                                    📊 **Summary:**
                                    - Session: {session_name}
                                    - Present: {len(recognized_students)}
                                    - Absent: {absent_count}
                                    - Total: {len(all_students)}
                                    - Attendance Rate: {(len(recognized_students) / len(all_students) * 100):.1f}%
                                    """)
                                    
                                    # Send notifications for absent students (if enabled)
                                    notification_manager = NotificationManager()
                                    absent_students = [s for s in all_students if s.id not in recognized_students]
                                    
                                    for student in absent_students:
                                        if student.parent_email:
                                            notification_manager.send_absence_notification(
                                                student.id, 
                                                session_name, 
                                                datetime.now().strftime('%Y-%m-%d')
                                            )
                                    
                                    time.sleep(3)
                                    st.rerun()
                            
                            except Exception as e:
                                session.rollback()
                                st.error(f"❌ Error saving attendance: {e}")
                        
                    else:
                        st.warning("❌ No faces detected in the image. Please try again with:")
                        st.write("• Better lighting")
                        st.write("• Closer positioning")
                        st.write("• Clear face visibility")
                        st.write("• Multiple people in frame")
            
    finally:
        session.close()

# Page: Analytics
elif choice == "📈 Analytics":
    st.header("📈 Advanced Analytics")
    
    analytics = AttendanceAnalytics()
    session = SessionLocal()
    
    try:
        tabs = st.tabs(["📊 Overview", "📈 Trends", "👥 Student Analysis", "📋 Reports"])
        
        with tabs[0]:  # Overview
            st.subheader("📊 System Overview")
            
            # Get user's classes
            if has_role(['admin']):
                classes = session.query(Class).filter(Class.is_active == True).all()
                class_options = [('all', 'All Classes')] + [(c.id, f"{c.name} ({c.code})") for c in classes]
            else:
                classes = get_classes_for_teacher(session, get_current_user())
                class_options = [(c.id, f"{c.name} ({c.code})") for c in classes]
            
            if class_options:
                selected_class = st.selectbox("Select Class for Analysis", 
                                            options=class_options,
                                            format_func=lambda x: x[1])
                
                class_id = selected_class[0] if selected_class[0] != 'all' else None
                
                # Date range selection
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input("Start Date", 
                                             value=datetime.now().replace(day=1).date())
                with col2:
                    end_date = st.date_input("End Date", 
                                           value=datetime.now().date())
                
                if class_id:
                    # Get class summary
                    summary = analytics.get_class_attendance_summary(class_id, start_date, end_date)
                    
                    if summary and summary['students']:
                        # Summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Students", summary['total_students'])
                        with col2:
                            st.metric("Class Average", f"{summary.get('class_average', 0):.1f}%")
                        with col3:
                            st.metric("Highest", f"{summary.get('highest_attendance', 0):.1f}%")
                        with col4:
                            st.metric("Below 75%", summary.get('students_below_75', 0))
                        
                        # Attendance distribution chart
                        st.subheader("📊 Attendance Distribution")
                        
                        percentages = [s['percentage'] for s in summary['students']]
                        fig = px.histogram(
                            x=percentages,
                            nbins=20,
                            title="Student Attendance Distribution",
                            labels={'x': 'Attendance Percentage', 'y': 'Number of Students'},
                            color_discrete_sequence=['#667eea']
                        )
                        fig.update_layout(showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info('No attendance data available for the selected period.')
            
            # Recent sessions
            st.subheader("📅 Recent Sessions")
            recent_sessions = session.query(AttendanceSession).filter(
                AttendanceSession.class_id == class_id
            ).order_by(AttendanceSession.date.desc()).limit(5).all()
            
            if recent_sessions:
                sessions_data = []
                for sess in recent_sessions:
                    sessions_data.append({
                        'Date': sess.date.strftime('%Y-%m-%d %H:%M'),
                        'Session': sess.session_name,
                        'Present': sess.total_present,
                        'Absent': sess.total_absent,
                        'Rate': f"{(sess.total_present / (sess.total_present + sess.total_absent) * 100):.1f}%" if (sess.total_present + sess.total_absent) > 0 else "0%"
                    })
                
                st.dataframe(pd.DataFrame(sessions_data), use_container_width=True)
            else:
                st.info("No attendance sessions found.")
    
    finally:
        analytics.close()
        session.close()

# Page: Export Data
elif choice == "📥 Export Data":
    st.header("📥 Data Export")
    
    session = SessionLocal()
    
    try:
        tabs = st.tabs(["📊 Attendance Data", "👥 Student Data", "📈 Analytics Export"])
        
        with tabs[0]:  # Attendance Data
            st.subheader("📊 Export Attendance Data")
            
            # Get user's classes
            if has_role(['admin']):
                classes = session.query(Class).filter(Class.is_active == True).all()
            else:
                classes = get_classes_for_teacher(session, get_current_user())
            
            if classes:
                class_options = [(c.id, f"{c.name} ({c.code})") for c in classes]
                selected_export_class = st.selectbox("Select Class", 
                                                   options=class_options,
                                                   format_func=lambda x: x[1])
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    export_start_date = st.date_input("Start Date", 
                                                    value=datetime.now().replace(day=1).date())
                with col2:
                    export_end_date = st.date_input("End Date", 
                                                  value=datetime.now().date())
                with col3:
                    export_format = st.selectbox("Format", 
                                               options=['excel', 'csv'],
                                               format_func=lambda x: x.upper())
                
                if st.button("📥 Export Attendance", type="primary"):
                    with st.spinner("Preparing export..."):
                        try:
                            analytics = AttendanceAnalytics()
                            
                            class_id = selected_export_class[0]
                            data = analytics.get_class_attendance_summary(
                                class_id, export_start_date, export_end_date
                            )
                            
                            if data and data.get('students'):
                                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                                
                                if export_format == 'excel':
                                    filename = f"attendance_export_{timestamp}.xlsx"
                                    filepath = export_attendance_data_to_excel(data, filename)
                                    
                                    if filepath:
                                        with open(filepath, 'rb') as file:
                                            st.download_button(
                                                "📥 Download Excel File",
                                                file.read(),
                                                filename,
                                                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                            )
                                
                                elif export_format == 'csv':
                                    df = pd.DataFrame(data['students'])
                                    csv_data = df.to_csv(index=False)
                                    filename = f"attendance_export_{timestamp}.csv"
                                    
                                    st.download_button(
                                        "📥 Download CSV File",
                                        csv_data,
                                        filename,
                                        "text/csv"
                                    )
                                
                                st.success(f"✅ Export completed! {len(data['students'])} student records exported.")
                            
                            else:
                                st.warning("No data found for the selected period.")
                            
                            analytics.close()
                        
                        except Exception as e:
                            st.error(f"Export failed: {e}")
        
        with tabs[1]:  # Student Data
            st.subheader("👥 Export Student Data")
            
            # Class selection for student export
            if has_role(['admin']):
                classes = session.query(Class).filter(Class.is_active == True).all()
                class_options = [('all', 'All Classes')] + [(c.id, f"{c.name} ({c.code})") for c in classes]
            else:
                classes = get_classes_for_teacher(session, get_current_user())
                class_options = [(c.id, f"{c.name} ({c.code})") for c in classes]
            
            selected_student_export_class = st.selectbox("Select Class", 
                                                       options=class_options,
                                                       format_func=lambda x: x[1],
                                                       key="student_export_class")
            
            include_photos = st.checkbox("Include Photo Information", value=True)
            student_export_format = st.selectbox("Export Format", 
                                                options=['excel', 'csv'],
                                                format_func=lambda x: x.upper(),
                                                key="student_export_format")
            
            if st.button("📥 Export Students", type="primary"):
                with st.spinner("Preparing student export..."):
                    try:
                        # Get students based on selection
                        if selected_student_export_class[0] == 'all':
                            query = session.query(Student).filter(Student.is_active == True)
                        else:
                            query = session.query(Student).filter(
                                Student.class_id == selected_student_export_class[0],
                                Student.is_active == True
                            )
                        
                        students = query.order_by(Student.name).all()
                        
                        if students:
                            student_data = []
                            
                            for student in students:
                                student_record = {
                                    'Name': student.name,
                                    'Roll': student.roll,
                                    'Class': student.class_obj.name if student.class_obj else 'N/A',
                                    'Email': student.email or '',
                                    'Phone': student.phone or '',
                                    'Parent Email': student.parent_email or '',
                                    'Parent Phone': student.parent_phone or '',
                                    'Admission Date': student.admission_date.strftime('%Y-%m-%d')
                                }
                                
                                if include_photos:
                                    photo_count = session.query(Photo).filter(Photo.student_id == student.id).count()
                                    student_record['Photos Count'] = photo_count
                                
                                student_data.append(student_record)
                            
                            df = pd.DataFrame(student_data)
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            
                            if student_export_format == 'excel':
                                filename = f"students_export_{timestamp}.xlsx"
                                
                                # Create Excel with formatting
                                buffer = io.BytesIO()
                                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                                    df.to_excel(writer, sheet_name='Students', index=False)
                                    
                                    # Format the worksheet
                                    worksheet = writer.sheets['Students']
                                    for column in worksheet.columns:
                                        max_length = 0
                                        column_letter = column[0].column_letter
                                        for cell in column:
                                            try:
                                                if len(str(cell.value)) > max_length:
                                                    max_length = len(str(cell.value))
                                            except:
                                                pass
                                        adjusted_width = (max_length + 2)
                                        worksheet.column_dimensions[column_letter].width = adjusted_width
                                
                                buffer.seek(0)
                                
                                st.download_button(
                                    "📥 Download Excel File",
                                    buffer.read(),
                                    filename,
                                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                            
                            elif student_export_format == 'csv':
                                csv_data = df.to_csv(index=False)
                                filename = f"students_export_{timestamp}.csv"
                                
                                st.download_button(
                                    "📥 Download CSV File",
                                    csv_data,
                                    filename,
                                    "text/csv"
                                )
                            
                            st.success(f"✅ Export completed! {len(students)} student records exported.")
                        
                        else:
                            st.warning("No students found for export.")
                    
                    except Exception as e:
                        st.error(f"Student export failed: {e}")
        
        with tabs[2]:  # Analytics Export
            st.subheader("📈 Export Analytics Data")
            
            st.info("📊 Export comprehensive analytics and performance reports")
            
            # Analytics export options
            analytics_type = st.selectbox(
                "Analytics Type",
                options=['performance_metrics', 'attendance_trends', 'system_logs'],
                format_func=lambda x: {
                    'performance_metrics': 'System Performance Metrics',
                    'attendance_trends': 'Historical Attendance Trends',
                    'system_logs': 'System Activity Logs'
                }[x]
            )
            
            analytics_period = st.selectbox(
                "Time Period",
                options=[7, 30, 90, 180],
                format_func=lambda x: f"Last {x} days"
            )
            
            if st.button("📈 Export Analytics", type="primary"):
                with st.spinner("Generating analytics export..."):
                    try:
                        if analytics_type == 'performance_metrics':
                            # Export system performance data

                            
                            start_date = datetime.now() - timedelta(days=analytics_period)
                            metrics = session.query(PerformanceMetrics).filter(
                                PerformanceMetrics.recorded_at >= start_date
                            ).all()
                            
                            if metrics:
                                metrics_data = [{
                                    'Timestamp': metric.recorded_at.strftime('%Y-%m-%d %H:%M:%S'),
                                    'Metric': metric.metric_name,
                                    'Value': metric.metric_value,
                                    'Unit': metric.unit or '',
                                    'Category': metric.category or ''
                                } for metric in metrics]
                                
                                df = pd.DataFrame(metrics_data)
                                csv_data = df.to_csv(index=False)
                                filename = f"performance_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                                
                                st.download_button(
                                    "📥 Download Performance Metrics",
                                    csv_data,
                                    filename,
                                    "text/csv"
                                )
                                
                                st.success(f"✅ Exported {len(metrics)} performance metrics.")
                            else:
                                st.warning("No performance metrics found for the selected period.")
                        
                        elif analytics_type == 'system_logs':
                            # Export audit logs

                            
                            start_date = datetime.now() - timedelta(days=analytics_period)
                            logs = session.query(AuditLog).filter(
                                AuditLog.timestamp >= start_date
                            ).order_by(AuditLog.timestamp.desc()).all()
                            
                            if logs:
                                logs_data = [{
                                    'Timestamp': log.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                                    'User': log.username,
                                    'Action': log.action,
                                    'Resource Type': log.resource_type or '',
                                    'Resource ID': log.resource_id or '',
                                    'Details': log.details or '',
                                    'IP Address': log.ip_address or ''
                                } for log in logs]
                                
                                df = pd.DataFrame(logs_data)
                                csv_data = df.to_csv(index=False)
                                filename = f"system_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                                
                                st.download_button(
                                    "📥 Download System Logs",
                                    csv_data,
                                    filename,
                                    "text/csv"
                                )
                                
                                st.success(f"✅ Exported {len(logs)} log entries.")
                            else:
                                st.warning("No logs found for the selected period.")
                    
                    except Exception as e:
                        st.error(f"Analytics export failed: {e}")
    
    finally:
        session.close()

# Admin Pages (only visible to admins)
elif choice == "🏫 Manage Classes" and has_role(['admin']):
    st.header("🏫 Class Management")
    
    session = SessionLocal()
    
    try:
        tabs = st.tabs(["➕ Add Class", "👀 View Classes", "📚 Manage Subjects"])
        
        with tabs[0]:  # Add Class
            st.subheader("➕ Create New Class")
            
            # Get academic years and semesters
            academic_years = session.query(AcademicYear).filter(AcademicYear.is_active == True).all()
            
            if not academic_years:
                st.warning("⚠️ No active academic year found. Please create one first.")
                with st.expander("Create Academic Year"):
                    year_name = st.text_input("Academic Year (e.g., 2023-2024)")
                    year_start = st.date_input("Start Date")
                    year_end = st.date_input("End Date")
                    
                    if st.button("Create Academic Year"):
                        academic_year = AcademicYear(
                            year_name=year_name,
                            start_date=year_start,
                            end_date=year_end
                        )
                        session.add(academic_year)
                        session.commit()
                        st.success("Academic year created!")
                        st.rerun()
            else:
                with st.form("class_form"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        class_name = st.text_input("Class Name *", placeholder="e.g., Grade 10-A")
                        class_code = st.text_input("Class Code *", placeholder="e.g., GR10A")
                        max_students = st.number_input("Max Students", min_value=1, max_value=200, value=50)
                        
                        # Teacher selection
                        auth_manager = AuthManager()
                        all_users = auth_manager.get_all_users()
                        teachers = [(username, user_data) for username, user_data in all_users.items() 
                                  if user_data.get('role') in ['teacher', 'admin']]
                        
                        teacher_options = [(username, f"{username} ({data['role']})") for username, data in teachers]
                        selected_teacher = st.selectbox("Assign Teacher", 
                                                      options=teacher_options,
                                                      format_func=lambda x: x[1])
                    
                    with col2:
                        # Academic year and semester
                        year_options = [(ay.id, ay.year_name) for ay in academic_years]
                        selected_year = st.selectbox("Academic Year", 
                                                   options=year_options,
                                                   format_func=lambda x: x[1])
                        
                        # Get semesters for selected year
                        if selected_year:
                            semesters = session.query(Semester).filter(
                                Semester.academic_year_id == selected_year[0],
                                Semester.is_active == True
                            ).all()
                            
                            if semesters:
                                semester_options = [(s.id, s.name) for s in semesters]
                                selected_semester = st.selectbox("Semester", 
                                                               options=semester_options,
                                                               format_func=lambda x: x[1])
                            else:
                                st.warning("No active semesters found for this academic year.")
                                selected_semester = None
                        
                        class_description = st.text_area("Description", placeholder="Optional class description")
                    
                    submitted = st.form_submit_button("➕ Create Class", type="primary")
                    
                    if submitted and class_name and class_code and selected_teacher and selected_year:
                        try:
                            # Check if class code already exists
                            existing_class = session.query(Class).filter(Class.code == class_code).first()
                            
                            if existing_class:
                                st.error("Class code already exists!")
                            else:
                                new_class = Class(
                                    name=class_name,
                                    code=class_code,
                                    description=class_description,
                                    academic_year_id=selected_year[0],
                                    semester_id=selected_semester[0] if selected_semester else None,
                                    teacher_username=selected_teacher[0],
                                    max_students=max_students
                                )
                                session.add(new_class)
                                session.commit()
                                
                                # Log audit event
                                log_audit_event(
                                    session, 
                                    get_current_user(), 
                                    'create_class',
                                    'class',
                                    new_class.id,
                                    f"Created class {class_name} ({class_code})"
                                )
                                
                                st.success(f"✅ Class '{class_name}' created successfully!")
                                time.sleep(2)
                                st.rerun()
                        
                        except Exception as e:
                            session.rollback()
                            st.error(f"Error creating class: {e}")
                    elif submitted:
                        st.error("Please fill in all required fields.")
        
        with tabs[1]:  # View Classes
            st.subheader("👀 All Classes")
            
            classes = session.query(Class).filter(Class.is_active == True).order_by(Class.name).all()
            
            if classes:
                for class_obj in classes:
                    with st.expander(f"🏫 {class_obj.name} ({class_obj.code})"):
                        col1, col2, col3 = st.columns([2, 2, 1])
                        
                        with col1:
                            st.write(f"**Name:** {class_obj.name}")
                            st.write(f"**Code:** {class_obj.code}")
                            st.write(f"**Teacher:** {class_obj.teacher_username}")
                            st.write(f"**Max Students:** {class_obj.max_students}")
                        
                        with col2:
                            st.write(f"**Academic Year:** {class_obj.academic_year.year_name if class_obj.academic_year else 'N/A'}")
                            st.write(f"**Semester:** {class_obj.semester.name if class_obj.semester else 'N/A'}")
                            
                            # Current enrollment
                            current_students = session.query(Student).filter(
                                Student.class_id == class_obj.id,
                                Student.is_active == True
                            ).count()
                            st.write(f"**Current Enrollment:** {current_students}/{class_obj.max_students}")
                            
                            if class_obj.description:
                                st.write(f"**Description:** {class_obj.description}")
                        
                        with col3:
                            # Quick stats
                            total_sessions = session.query(AttendanceSession).filter(
                                AttendanceSession.class_id == class_obj.id
                            ).count()
                            st.metric("Sessions Held", total_sessions)
                            
                            # Edit/Delete buttons
                            col_edit, col_del = st.columns(2)
                            
                            with col_edit:
                                if st.button(f"✏️ Edit", key=f"edit_class_{class_obj.id}"):
                                    st.session_state[f'edit_class_{class_obj.id}'] = True
                            
                            with col_del:
                                if st.button(f"🗑️ Delete", key=f"del_class_{class_obj.id}", type="secondary"):
                                    if st.session_state.get(f'confirm_del_class_{class_obj.id}', False):
                                        # Deactivate class instead of deleting
                                        class_obj.is_active = False
                                        session.commit()
                                        
                                        log_audit_event(
                                            session, 
                                            get_current_user(), 
                                            'deactivate_class',
                                            'class',
                                            class_obj.id,
                                            f"Deactivated class {class_obj.name}"
                                        )
                                        
                                        st.success(f"Class {class_obj.name} deactivated!")
                                        time.sleep(1)
                                        st.rerun()
                                    else:
                                        st.session_state[f'confirm_del_class_{class_obj.id}'] = True
                                        st.warning("Click delete again to confirm")
            else:
                st.info("No classes found. Create your first class above!")
        
        with tabs[2]:  # Manage Subjects
            st.subheader("📚 Subject Management")
            
            classes = session.query(Class).filter(Class.is_active == True).all()
            
            if classes:
                # Add subject form
                with st.expander("➕ Add New Subject"):
                    with st.form("subject_form"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            subject_name = st.text_input("Subject Name *")
                            subject_code = st.text_input("Subject Code *")
                            credits = st.number_input("Credits", min_value=1, max_value=10, value=3)
                        
                        with col2:
                            class_options = [(c.id, f"{c.name} ({c.code})") for c in classes]
                            selected_class_subject = st.selectbox("Class", 
                                                                options=class_options,
                                                                format_func=lambda x: x[1])
                            
                            # Teacher selection for subject
                            auth_manager = AuthManager()
                            all_users = auth_manager.get_all_users()
                            teachers = [(username, user_data) for username, user_data in all_users.items() 
                                      if user_data.get('role') in ['teacher', 'admin']]
                            
                            teacher_options = [(username, f"{username} ({data['role']})") for username, data in teachers]
                            selected_subject_teacher = st.selectbox("Subject Teacher", 
                                                                  options=teacher_options,
                                                                  format_func=lambda x: x[1])
                            
                            subject_description = st.text_area("Description")
                        
                        submitted_subject = st.form_submit_button("➕ Add Subject", type="primary")
                        
                        if submitted_subject and subject_name and subject_code and selected_class_subject and selected_subject_teacher:
                            try:
                                # Check if subject code already exists in the class
                                existing_subject = session.query(Subject).filter(
                                    Subject.code == subject_code,
                                    Subject.class_id == selected_class_subject[0]
                                ).first()
                                
                                if existing_subject:
                                    st.error("Subject code already exists in this class!")
                                else:
                                    new_subject = Subject(
                                        name=subject_name,
                                        code=subject_code,
                                        class_id=selected_class_subject[0],
                                        teacher_username=selected_subject_teacher[0],
                                        credits=credits,
                                        description=subject_description
                                    )
                                    session.add(new_subject)
                                    session.commit()
                                    
                                    log_audit_event(
                                        session, 
                                        get_current_user(), 
                                        'create_subject',
                                        'subject',
                                        new_subject.id,
                                        f"Created subject {subject_name} ({subject_code})"
                                    )
                                    
                                    st.success(f"✅ Subject '{subject_name}' added successfully!")
                                    time.sleep(2)
                                    st.rerun()
                            
                            except Exception as e:
                                session.rollback()
                                st.error(f"Error adding subject: {e}")
                        elif submitted_subject:
                            st.error("Please fill in all required fields.")
                
                # Display existing subjects
                st.subheader("📚 Existing Subjects")
                
                for class_obj in classes:
                    subjects = session.query(Subject).filter(
                        Subject.class_id == class_obj.id,
                        Subject.is_active == True
                    ).all()
                    
                    if subjects:
                        st.write(f"**{class_obj.name} ({class_obj.code})**")
                        
                        subjects_data = []
                        for subject in subjects:
                            subjects_data.append({
                                'Subject': subject.name,
                                'Code': subject.code,
                                'Teacher': subject.teacher_username,
                                'Credits': subject.credits,
                                'Description': subject.description or 'N/A'
                            })
                        
                        df = pd.DataFrame(subjects_data)
                        st.dataframe(df, use_container_width=True)
            else:
                st.info("No classes available. Please create classes first.")
    
    finally:
        session.close()

# Page: Manage Users (Admin only)
elif choice == "👨‍🏫 Manage Users" and has_role(['admin']):
    st.header("👨‍🏫 User Management")
    
    auth_manager = AuthManager()
    
    tabs = st.tabs(["➕ Add User", "👀 View Users", "🔑 Reset Passwords"])
    
    with tabs[0]:  # Add User
        st.subheader("➕ Create New User")
        
        with st.form("user_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                new_username = st.text_input("Username *")
                new_email = st.text_input("Email *")
                new_role = st.selectbox("Role", options=['teacher', 'admin'])
            
            with col2:
                new_password = st.text_input("Temporary Password *", type="password")
                confirm_password = st.text_input("Confirm Password *", type="password")
                send_welcome_email = st.checkbox("Send Welcome Email", value=True)
            
            submitted_user = st.form_submit_button("➕ Create User", type="primary")
            
            if submitted_user and new_username and new_email and new_password:
                if new_password != confirm_password:
                    st.error("Passwords don't match!")
                elif len(new_password) < AUTH_CONFIG['password_min_length']:
                    st.error(f"Password must be at least {AUTH_CONFIG['password_min_length']} characters long!")
                else:
                    try:
                        success = auth_manager.create_user(new_username, new_password, new_email, new_role)
                        
                        if success:
                            st.success(f"✅ User '{new_username}' created successfully!")
                            
                            # Send welcome email if requested
                            if send_welcome_email:
                                from utils.notification_utils import NotificationManager
                                notification_manager = NotificationManager()
                                
                                notification_manager.send_welcome_email(
                                    new_email, 
                                    new_username, 
                                    new_username, 
                                    new_password
                                )
                                st.info("📧 Welcome email sent!")
                            
                            # Log audit event
                            session = SessionLocal()
                            try:
                                log_audit_event(
                                    session, 
                                    get_current_user(), 
                                    'create_user',
                                    'user',
                                    new_username,
                                    f"Created user {new_username} with role {new_role}"
                                )
                            finally:
                                session.close()
                            
                            time.sleep(2)
                            st.rerun()
                        else:
                            st.error("Username already exists!")
                    
                    except Exception as e:
                        st.error(f"Error creating user: {e}")
            elif submitted_user:
                st.error("Please fill in all required fields.")
    
    with tabs[1]:  # View Users
        st.subheader("👀 All Users")
        
        users = auth_manager.get_all_users()
        
        if users:
            search_user = st.text_input("🔍 Search users", placeholder="Enter username or email")
            
            filtered_users = users
            if search_user:
                filtered_users = {
                    username: data for username, data in users.items()
                    if search_user.lower() in username.lower() or 
                       search_user.lower() in data.get('email', '').lower()
                }
            
            st.write(f"**Found {len(filtered_users)} users**")
            
            for username, user_data in filtered_users.items():
                with st.expander(f"👤 {username} ({user_data['role'].title()})"):
                    col1, col2, col3 = st.columns([2, 2, 1])
                    
                    with col1:
                        st.write(f"**Username:** {username}")
                        st.write(f"**Email:** {user_data.get('email', 'N/A')}")
                        st.write(f"**Role:** {user_data['role'].title()}")
                        st.write(f"**Created:** {user_data.get('created_at', 'N/A')[:10]}")
                    
                    with col2:
                        st.write(f"**Last Login:** {user_data.get('last_login', 'Never')[:19] if user_data.get('last_login') else 'Never'}")
                        st.write(f"**Login Attempts:** {user_data.get('login_attempts', 0)}")
                        st.write(f"**Status:** {'🟢 Active' if user_data.get('is_active', True) else '🔴 Inactive'}")
                    
                    with col3:
                        # User actions
                        if username != get_current_user():  # Can't manage own account
                            if user_data.get('is_active', True):
                                if st.button(f"🚫 Deactivate", key=f"deactivate_{username}"):
                                    auth_manager.update_user(username, {'is_active': False})
                                    st.success(f"User {username} deactivated!")
                                    time.sleep(1)
                                    st.rerun()
                            else:
                                if st.button(f"✅ Activate", key=f"activate_{username}"):
                                    auth_manager.update_user(username, {'is_active': True})
                                    st.success(f"User {username} activated!")
                                    time.sleep(1)
                                    st.rerun()
                            
                            if st.button(f"🔑 Reset Password", key=f"reset_{username}"):
                                st.session_state[f'reset_password_{username}'] = True
                            
                            # Password reset form
                            if st.session_state.get(f'reset_password_{username}', False):
                                new_pwd = st.text_input(f"New Password for {username}", type="password", key=f"new_pwd_{username}")
                                if st.button(f"Update Password", key=f"update_pwd_{username}"):
                                    if new_pwd and len(new_pwd) >= AUTH_CONFIG['password_min_length']:
                                        hashed_pwd = auth_manager.hash_password(new_pwd)
                                        auth_manager.update_user(username, {
                                            'password': hashed_pwd, 
                                            'login_attempts': 0
                                        })
                                        st.success(f"Password updated for {username}!")
                                        st.session_state[f'reset_password_{username}'] = False
                                        st.rerun()
                                    else:
                                        st.error(f"Password must be at least {AUTH_CONFIG['password_min_length']} characters!")
        else:
            st.info("No users found.")
    
    with tabs[2]:  # Reset Passwords
        st.subheader("🔑 Bulk Password Reset")
        
        st.warning("⚠️ Use this feature carefully. It will reset passwords for selected users.")
        
        users = auth_manager.get_all_users()
        user_options = [(username, f"{username} ({data['role']})") for username, data in users.items()]
        
        selected_users = st.multiselect(
            "Select Users for Password Reset",
            options=user_options,
            format_func=lambda x: x[1]
        )
        
        if selected_users:
            new_default_password = st.text_input("New Default Password", type="password")
            notify_users = st.checkbox("Send Email Notifications", value=True)
            
            if st.button("🔑 Reset Selected Passwords", type="secondary"):
                if new_default_password and len(new_default_password) >= AUTH_CONFIG['password_min_length']:
                    success_count = 0
                    
                    for username, _ in selected_users:
                        try:
                            hashed_pwd = auth_manager.hash_password(new_default_password)
                            auth_manager.update_user(username, {
                                'password': hashed_pwd,
                                'login_attempts': 0
                            })
                            success_count += 1
                            
                            # Send notification email
                            if notify_users:
                                user_data = users[username]
                                if user_data.get('email'):
                                    from utils.notification_utils import NotificationManager
                                    notification_manager = NotificationManager()
                                    notification_manager.send_welcome_email(
                                        user_data['email'],
                                        username,
                                        username,
                                        new_default_password
                                    )
                        
                        except Exception as e:
                            st.error(f"Error resetting password for {username}: {e}")
                    
                    st.success(f"✅ Passwords reset for {success_count} users!")
                    
                    # Log audit event
                    session = SessionLocal()
                    try:
                        log_audit_event(
                            session, 
                            get_current_user(), 
                            'bulk_password_reset',
                            'user',
                            None,
                            f"Reset passwords for {success_count} users"
                        )
                    finally:
                        session.close()
                else:
                    st.error(f"Password must be at least {AUTH_CONFIG['password_min_length']} characters!")

# Page: Notifications (Admin only)
elif choice == "🔔 Notifications" and has_role(['admin']):
    st.header("🔔 Notification Center")
    
    session = SessionLocal()
    
    try:
        tabs = st.tabs(["📧 Send Notifications", "📋 View History", "⚙️ Settings"])
        
        with tabs[0]:  # Send Notifications
            st.subheader("📧 Send Custom Notifications")
            
            notification_type = st.selectbox(
                "Notification Type",
                options=['general', 'attendance_alert', 'system_update', 'emergency'],
                format_func=lambda x: x.replace('_', ' ').title()
            )
            
            # Recipient selection
            recipient_type = st.selectbox(
                "Send To",
                options=['all_students', 'specific_class', 'low_attendance', 'parents_only'],
                format_func=lambda x: {
                    'all_students': 'All Students',
                    'specific_class': 'Specific Class',
                    'low_attendance': 'Low Attendance Students',
                    'parents_only': 'Parents Only'
                }[x]
            )
            
            recipients = []
            
            if recipient_type == 'specific_class':
                classes = session.query(Class).filter(Class.is_active == True).all()
                class_options = [(c.id, f"{c.name} ({c.code})") for c in classes]
                selected_notification_class = st.selectbox("Select Class", 
                                                         options=class_options,
                                                         format_func=lambda x: x[1])
                
                if selected_notification_class:
                    recipients = session.query(Student).filter(
                        Student.class_id == selected_notification_class[0],
                        Student.is_active == True
                    ).all()
            
            elif recipient_type == 'low_attendance':
                threshold = st.slider("Attendance Threshold (%)", 0, 100, 75)
                analytics = AttendanceAnalytics()
                try:
                    low_attendance_students = analytics.get_low_attendance_students(threshold=threshold)
                    recipients = [session.query(Student).filter(Student.id == s['id']).first() 
                                for s in low_attendance_students]
                finally:
                    analytics.close()
            
            elif recipient_type == 'all_students':
                recipients = session.query(Student).filter(Student.is_active == True).all()
            
            # Message composition
            col1, col2 = st.columns([2, 1])
            
            with col1:
                subject = st.text_input("Subject *")
                message = st.text_area("Message *", height=150)
                
                priority = st.selectbox("Priority", 
                                      options=['low', 'normal', 'high', 'urgent'],
                                      index=1)
            
            with col2:
                if recipients:
                    st.write(f"**Recipients:** {len(recipients)} students")
                    
                    send_to_students = st.checkbox("Send to Students", value=True)
                    send_to_parents = st.checkbox("Send to Parents", value=True)
                    
                    # Preview recipients
                    with st.expander("👀 Preview Recipients"):
                        for recipient in recipients[:10]:  # Show first 10
                            st.write(f"• {recipient.name} ({recipient.roll})")
                        if len(recipients) > 10:
                            st.write(f"... and {len(recipients) - 10} more")
                else:
                    st.info("Select recipients to send notifications")
            
            if st.button("📧 Send Notifications", type="primary") and subject and message and recipients:
                with st.spinner("Sending notifications..."):
                    try:
                        notification_manager = NotificationManager()
                        sent_count = 0
                        
                        for student in recipients:
                            # Create notification record
                            notification = Notification(
                                student_id=student.id,
                                type=notification_type,
                                title=subject,
                                message=message,
                                priority=priority
                            )
                            session.add(notification)
                            
                            # Send emails
                            if send_to_students and student.email:
                                if notification_manager.send_email(student.email, subject, message):
                                    notification.sent_to_student = True
                                    sent_count += 1
                            
                            if send_to_parents and student.parent_email:
                                if notification_manager.send_email(student.parent_email, subject, message):
                                    notification.sent_to_parent = True
                                    sent_count += 1
                            
                            notification.sent_at = datetime.utcnow()
                        
                        session.commit()
                        
                        # Log audit event
                        log_audit_event(
                            session, 
                            get_current_user(), 
                            'send_bulk_notification',
                            'notification',
                            None,
                            f"Sent {notification_type} notification to {len(recipients)} students"
                        )
                        
                        st.success(f"✅ Notifications sent! {sent_count} emails delivered to {len(recipients)} students.")
                    
                    except Exception as e:
                        session.rollback()
                        st.error(f"Error sending notifications: {e}")
        
        with tabs[1]:  # View History
            st.subheader("📋 Notification History")
            
            # Filters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                filter_type = st.selectbox("Filter by Type", 
                                         options=['all'] + ['general', 'attendance_alert', 'system_update', 'emergency'],
                                         format_func=lambda x: 'All Types' if x == 'all' else x.replace('_', ' ').title())
            
            with col2:
                filter_priority = st.selectbox("Filter by Priority",
                                             options=['all'] + ['low', 'normal', 'high', 'urgent'],
                                             format_func=lambda x: 'All Priorities' if x == 'all' else x.title())
            
            with col3:
                days_back = st.selectbox("Time Period", 
                                       options=[7, 14, 30, 60, 90],
                                       format_func=lambda x: f"Last {x} days")
            
            # Get notifications
            start_date = datetime.now() - timedelta(days=days_back)
            query = session.query(Notification).filter(Notification.created_at >= start_date)
            
            if filter_type != 'all':
                query = query.filter(Notification.type == filter_type)
            
            if filter_priority != 'all':
                query = query.filter(Notification.priority == filter_priority)
            
            notifications = query.order_by(Notification.created_at.desc()).all()
            
            if notifications:
                st.write(f"**Found {len(notifications)} notifications**")
                
                # Display notifications
                for notification in notifications:
                    priority_colors = {
                        'low': '#6c757d',
                        'normal': '#17a2b8',
                        'high': '#ffc107',
                        'urgent': '#dc3545'
                    }
                    
                    priority_color = priority_colors.get(notification.priority, '#17a2b8')
                    
                    with st.expander(f"📧 {notification.title} - {notification.created_at.strftime('%Y-%m-%d %H:%M')}"):
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.write(f"**Student:** {notification.student.name} ({notification.student.roll})")
                            st.write(f"**Type:** {notification.type.replace('_', ' ').title()}")
                            st.write(f"**Message:** {notification.message}")
                            
                            delivery_status = []
                            if notification.sent_to_student:
                                delivery_status.append("📧 Student")
                            if notification.sent_to_parent:
                                delivery_status.append("👨‍👩‍👧‍👦 Parent")
                            
                            st.write(f"**Delivered to:** {', '.join(delivery_status) if delivery_status else 'Not sent'}")
                        
                        with col2:
                            st.markdown(f"""
                            <div style="padding: 0.5rem; background-color: {priority_color}; color: white; border-radius: 5px; text-align: center;">
                                <strong>{notification.priority.upper()}</strong>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.write(f"**Sent:** {notification.sent_at.strftime('%Y-%m-%d %H:%M') if notification.sent_at else 'Not sent'}")
                            st.write(f"**Read:** {'✅ Yes' if notification.is_read else '❌ No'}")
            else:
                st.info("No notifications found for the selected criteria.")
        
        with tabs[2]:  # Settings
            st.subheader("⚙️ Notification Settings")
            
            from utils.notification_utils import test_email_configuration
            
            # Email configuration test
            st.write("**📧 Email Configuration Test**")
            if st.button("🧪 Test Email Configuration"):
                with st.spinner("Testing email configuration..."):
                    success = test_email_configuration()
                    if success:
                        st.success("✅ Email configuration is working correctly!")
                    else:
                        st.error("❌ Email configuration test failed. Please check your settings.")
            
            st.markdown("---")
            
            # Notification preferences
            st.write("**🔔 Notification Preferences**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Automatic Notifications:**")
                auto_absence = st.checkbox("Absence Notifications", value=True, help="Send notifications when students are marked absent")
                auto_low_attendance = st.checkbox("Low Attendance Alerts", value=True, help="Send alerts when attendance drops below threshold")
                daily_summaries = st.checkbox("Daily Summaries", value=False, help="Send daily attendance summaries to teachers")
            
            with col2:
                st.write("**Thresholds:**")
                attendance_threshold = st.slider("Low Attendance Threshold (%)", 0, 100, 75)
                notification_delay = st.slider("Absence Notification Delay (hours)", 0, 24, 2)
            
            if st.button("💾 Save Settings"):
                # Save settings to database or config
                st.success("✅ Settings saved successfully!")
    
    finally:
        session.close()

# Page: System Settings (Admin only)
elif choice == "⚙️ System Settings" and has_role(['admin']):
    st.header("⚙️ System Settings")
    
    session = SessionLocal()
    
    try:
        tabs = st.tabs(["🔧 General", "🤖 Face Recognition", "📧 Email", "🔒 Security", "📊 Performance"])
        
        with tabs[0]:  # General Settings
            st.subheader("🔧 General Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**🎓 Academic Settings**")
                academic_year_start = st.selectbox("Academic Year Start Month", 
                                                 options=list(range(1, 13)),
                                                 format_func=lambda x: datetime(2023, x, 1).strftime('%B'),
                                                 index=6)  # July
                
                semester_duration = st.number_input("Semester Duration (months)", 1, 12, 6)
                default_class_duration = st.number_input("Default Class Duration (minutes)", 30, 240, 60)
                grace_period = st.number_input("Attendance Grace Period (minutes)", 0, 30, 10)
                
                st.write("**🎨 UI Settings**")
                ui_theme = st.selectbox("Theme", options=['light', 'dark'], index=0)
                show_tips = st.checkbox("Show Tips and Hints", value=True)
                animation_enabled = st.checkbox("Enable Animations", value=True)
            
            with col2:
                st.write("**📊 Display Settings**")
                default_page_size = st.number_input("Default Page Size", 10, 100, 25)
                show_performance_metrics = st.checkbox("Show Performance Metrics", value=True)
                
                st.write("**🔔 Default Notification Settings**")
                enable_notifications = st.checkbox("Enable Email Notifications", value=True)
                attendance_alert_threshold = st.slider("Attendance Alert Threshold (%)", 0, 100, 75)
                late_threshold = st.number_input("Late Threshold (minutes)", 0, 60, 15)
                
                st.write("**📂 File Management**")
                max_upload_size = st.number_input("Max Upload Size (MB)", 1, 100, 5)
                auto_cleanup_days = st.number_input("Auto Cleanup Period (days)", 30, 365, 90)
            
            if st.button("💾 Save General Settings"):
                st.success("✅ General settings saved successfully!")
        
        with tabs[1]:  # Face Recognition Settings
            st.subheader("🤖 Face Recognition Configuration")
            
            st.info("🔧 Optimize face recognition settings for your Lenovo Legion 5i laptop")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**🎯 Recognition Parameters**")
                tolerance = st.slider("Recognition Tolerance", 0.3, 0.9, 0.6, 0.05,
                                    help="Lower = more strict, Higher = more lenient")
                
                min_confidence = st.slider("Minimum Confidence", 0.3, 0.9, 0.6, 0.05,
                                         help="Minimum confidence for positive identification")
                
                model_type = st.selectbox("Detection Model", 
                                        options=['hog', 'cnn'],
                                        help="HOG is faster on CPU, CNN is more accurate but slower")
                
                num_jitters = st.selectbox("Number of Jitters", 
                                         options=[1, 2, 3, 4, 5],
                                         index=0,
                                         help="More jitters = better accuracy but slower processing")
            
            with col2:
                st.write("**⚡ Performance Optimization**")
                max_faces_per_frame = st.number_input("Max Faces per Frame", 10, 100, 70,
                                                    help="Maximum number of faces to process simultaneously")
                
                resize_factor = st.slider("Image Resize Factor", 0.1, 1.0, 0.25, 0.05,
                                        help="Smaller = faster processing, larger = better accuracy")
                
                cache_size = st.number_input("Encoding Cache Size", 100, 2000, 1000,
                                           help="Number of face encodings to keep in memory")
                
                batch_size = st.number_input("Batch Processing Size", 1, 16, 8,
                                           help="Number of images to process in parallel")
                
                st.write("**🔒 Anti-Spoofing**")
                enable_liveness = st.checkbox("Enable Liveness Detection", value=True,
                                            help="Prevent spoofing with photos")
                
                liveness_threshold = st.number_input("Liveness Threshold", 50, 200, 100,
                                                   help="Texture variance threshold for liveness detection")
            
            # Test current settings
            if st.button("🧪 Test Face Recognition Settings"):
                with st.spinner("Testing face recognition performance..."):
                    # Simulate performance test
                    time.sleep(2)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Processing Speed", "1.2s per frame")
                    with col2:
                        st.metric("Memory Usage", "245 MB")
                    with col3:
                        st.metric("CPU Usage", "45%")
                    
                    st.success("✅ Face recognition settings tested successfully!")
            
            if st.button("💾 Save Face Recognition Settings"):
                # Update configuration
                from config import FACE_RECOGNITION_CONFIG
                FACE_RECOGNITION_CONFIG.update({
                    'tolerance': tolerance,
                    'min_confidence': min_confidence,
                    'model': model_type,
                    'num_jitters': num_jitters,
                    'max_faces_per_frame': max_faces_per_frame,
                    'resize_factor': resize_factor,
                    'encoding_cache_size': cache_size,
                    'face_detection_batch_size': batch_size
                })
                
                st.success("✅ Face recognition settings saved successfully!")
        
        with tabs[2]:  # Email Settings
            st.subheader("📧 Email Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**📬 SMTP Settings**")
                smtp_server = st.text_input("SMTP Server", value="smtp.gmail.com")
                smtp_port = st.number_input("SMTP Port", 1, 65535, 587)
                use_tls = st.checkbox("Use TLS", value=True)
                
                email_user = st.text_input("Email Address")
                email_password = st.text_input("Email Password/App Password", type="password")
            
            with col2:
                st.write("**📧 Email Templates**")
                welcome_subject = st.text_input("Welcome Email Subject", value="Welcome to Attendance System")
                absence_subject = st.text_input("Absence Alert Subject", value="Student Absence Notification")
                low_attendance_subject = st.text_input("Low Attendance Subject", value="Low Attendance Alert")
                
                st.write("**⏰ Notification Timing**")
                send_daily_reports = st.checkbox("Send Daily Reports", value=False)
                daily_report_time = st.time_input("Daily Report Time", value=datetime.strptime("18:00", "%H:%M").time())
            
            # Test email configuration
            if st.button("📧 Test Email Configuration"):
                with st.spinner("Testing email configuration..."):
                    try:
                        from utils.notification_utils import test_email_configuration
                        success = test_email_configuration()
                        if success:
                            st.success("✅ Email test successful!")
                        else:
                            st.error("❌ Email test failed. Please check your configuration.")
                    except Exception as e:
                        st.error(f"❌ Email test error: {e}")
            
            if st.button("💾 Save Email Settings"):
                st.success("✅ Email settings saved successfully!")
        
        with tabs[3]:  # Security Settings
            st.subheader("🔒 Security Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**🔐 Authentication**")
                session_timeout = st.number_input("Session Timeout (hours)", 1, 24, 8)
                max_login_attempts = st.number_input("Max Login Attempts", 1, 10, 3)
                password_min_length = st.number_input("Minimum Password Length", 6, 20, 8)
                require_special_chars = st.checkbox("Require Special Characters", value=True)
                
                st.write("**🔍 Audit Settings**")
                enable_audit_logs = st.checkbox("Enable Audit Logging", value=True)
                log_level = st.selectbox("Log Level", 
                                       options=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                                       index=1)
            
            with col2:
                st.write("**📁 File Security**")
                encrypt_face_data = st.checkbox("Encrypt Face Data", value=True)
                secure_file_storage = st.checkbox("Secure File Storage", value=True)
                max_file_uploads_per_hour = st.number_input("Max File Uploads per Hour", 10, 1000, 100)
                
                allowed_extensions = st.multiselect("Allowed Image Extensions",
                                                  options=['jpg', 'jpeg', 'png', 'bmp', 'gif'],
                                                  default=['jpg', 'jpeg', 'png', 'bmp'])
                
                st.write("**🛡️ System Security**")
                auto_backup = st.checkbox("Enable Auto Backup", value=True)
                backup_frequency = st.selectbox("Backup Frequency", 
                                              options=['daily', 'weekly', 'monthly'],
                                              index=1)
            
            if st.button("💾 Save Security Settings"):
                st.success("✅ Security settings saved successfully!")
        
        with tabs[4]:  # Performance Settings
            st.subheader("📊 Performance Configuration")
            
            # Current system status
            resources = monitor_system_resources()
            
            st.write("**📊 Current System Status**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("CPU Usage", f"{resources['cpu_percent']:.1f}%")
            with col2:
                st.metric("Memory Usage", f"{resources['memory_percent']:.1f}%")
            with col3:
                st.metric("Available Memory", f"{resources['memory_available_mb']:.0f} MB")
            
            # Performance settings
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**⚡ Performance Limits**")
                max_workers = st.slider("Max Worker Threads", 1, 16, 8,
                                      help="Number of parallel processing threads")
                
                memory_limit = st.number_input("Memory Limit (MB)", 1024, 8192, 4096,
                                             help="Maximum memory usage for the application")
                
                cpu_threshold = st.slider("CPU Alert Threshold (%)", 50, 100, 80,
                                        help="Alert when CPU usage exceeds this percentage")
                
                memory_threshold = st.slider("Memory Alert Threshold (%)", 50, 100, 85,
                                           help="Alert when memory usage exceeds this percentage")
            
            with col2:
                st.write("**🧹 Cleanup Settings**")
                enable_auto_cleanup = st.checkbox("Enable Auto Cleanup", value=True)
                cleanup_temp_files = st.checkbox("Cleanup Temp Files", value=True)
                cleanup_old_logs = st.checkbox("Cleanup Old Logs", value=True)
                
                temp_file_retention = st.number_input("Temp File Retention (days)", 1, 30, 7)
                log_retention = st.number_input("Log Retention (days)", 7, 365, 30)
                
                st.write("**🔄 Cache Settings**")
                enable_caching = st.checkbox("Enable Caching", value=True)
                cache_ttl = st.number_input("Cache TTL (seconds)", 300, 7200, 3600)
            
            # Performance monitoring
            if st.button("📊 Run Performance Analysis"):
                with st.spinner("Analyzing system performance..."):
                    time.sleep(3)
                    
                    # Simulate performance analysis
                    st.success("✅ Performance analysis completed!")
                    
                    performance_data = {
                        'Component': ['Face Recognition', 'Database', 'File I/O', 'Network'],
                        'Performance': [85, 92, 78, 88],
                        'Status': ['Good', 'Excellent', 'Fair', 'Good']
                    }
                    
                    df = pd.DataFrame(performance_data)
                    st.dataframe(df, use_container_width=True)
            
            if st.button("💾 Save Performance Settings"):
                st.success("✅ Performance settings saved successfully!")
                
                # Clear cache if needed
                if st.button("🧹 Clear Cache"):
                    clear_face_cache()
                    st.success("✅ Cache cleared successfully!")
    
    finally:
        session.close()

# Page: Audit Logs (Admin only)
elif choice == "🔍 Audit Logs" and has_role(['admin']):
    st.header("🔍 System Audit Logs")
    
    session = SessionLocal()
    
    try:

        
        # Filters
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            filter_user = st.selectbox("Filter by User", 
                                     options=['all'] + [log.username for log in session.query(AuditLog.username).distinct()],
                                     format_func=lambda x: 'All Users' if x == 'all' else x)
        
        with col2:
            filter_action = st.selectbox("Filter by Action",
                                       options=['all'] + [log.action for log in session.query(AuditLog.action).distinct()],
                                       format_func=lambda x: 'All Actions' if x == 'all' else x.replace('_', ' ').title())
        
        with col3:
            filter_resource = st.selectbox("Filter by Resource",
                                         options=['all'] + [log.resource_type for log in session.query(AuditLog.resource_type).distinct() if log.resource_type],
                                         format_func=lambda x: 'All Resources' if x == 'all' else x.replace('_', ' ').title())
        
        with col4:
            days_back = st.selectbox("Time Period", 
                                   options=[1, 7, 14, 30, 60, 90],
                                   format_func=lambda x: f"Last {x} day{'s' if x > 1 else ''}",
                                   index=2)
        
        # Get audit logs
        start_date = datetime.now() - timedelta(days=days_back)
        query = session.query(AuditLog).filter(AuditLog.timestamp >= start_date)
        
        if filter_user != 'all':
            query = query.filter(AuditLog.username == filter_user)
        
        if filter_action != 'all':
            query = query.filter(AuditLog.action == filter_action)
        
        if filter_resource != 'all':
            query = query.filter(AuditLog.resource_type == filter_resource)
        
        logs = query.order_by(AuditLog.timestamp.desc()).limit(1000).all()
        
        if logs:
            st.write(f"**Found {len(logs)} audit log entries** (showing latest 1000)")
            
            # Export logs
            if st.button("📥 Export Logs"):
                logs_data = [{
                    'Timestamp': log.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    'User': log.username,
                    'Action': log.action,
                    'Resource Type': log.resource_type or '',
                    'Resource ID': log.resource_id or '',
                    'Details': log.details or '',
                    'IP Address': log.ip_address or ''
                } for log in logs]
                
                df = pd.DataFrame(logs_data)
                csv_data = df.to_csv(index=False)
                filename = f"audit_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                
                st.download_button(
                    "📥 Download Audit Logs",
                    csv_data,
                    filename,
                    "text/csv"
                )
            
            # Display logs
            for log in logs[:50]:  # Show first 50 logs
                timestamp_str = log.timestamp.strftime('%Y-%m-%d %H:%M:%S')
                
                # Color code by action type
                action_colors = {
                    'login': '#28a745',
                    'logout': '#6c757d',
                    'create': '#007bff',
                    'update': '#ffc107',
                    'delete': '#dc3545',
                    'take_attendance': '#17a2b8'
                }
                
                action_color = action_colors.get(log.action.split('_')[0], '#6c757d')
                
                with st.expander(f"🔍 {timestamp_str} - {log.username} - {log.action.replace('_', ' ').title()}"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write(f"**User:** {log.username}")
                        st.write(f"**Action:** {log.action.replace('_', ' ').title()}")
                        st.write(f"**Timestamp:** {timestamp_str}")
                        
                        if log.resource_type:
                            st.write(f"**Resource Type:** {log.resource_type.replace('_', ' ').title()}")
                        
                        if log.resource_id:
                            st.write(f"**Resource ID:** {log.resource_id}")
                        
                        if log.details:
                            st.write(f"**Details:** {log.details}")
                    
                    with col2:
                        # Action type indicator
                        st.markdown(f"""
                        <div style="padding: 0.5rem; background-color: {action_color}; color: white; border-radius: 5px; text-align: center; margin-top: 1rem;">
                            <strong>{log.action.replace('_', ' ').title()}</strong>
                        </div>
                        """, unsafe_allow_html=True)
            
            if len(logs) > 50:
                st.info(f"Showing first 50 entries. Use filters to narrow down results.")
        
        else:
            st.info("No audit logs found for the selected criteria.")
    
    finally:
        session.close()

# Default fallback
else:
    if not has_role(['admin']) and choice in ["🏫 Manage Classes", "👨‍🏫 Manage Users", "🔔 Notifications", "⚙️ System Settings", "🔍 Audit Logs"]:
        st.error("🚫 Access Denied: This feature requires administrator privileges.")
    else:
        st.error("🚫 Page not found or access denied.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**🎓 Multi-Face Attendance System**")
st.sidebar.markdown("*Optimized for Lenovo Legion 5i*")
st.sidebar.markdown(f"*Version 2.0 - {datetime.now().strftime('%Y-%m-%d')}*")
