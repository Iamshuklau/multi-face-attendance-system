import streamlit as st
import cv2
import pandas as pd
import os
from datetime import datetime
import numpy as np
from PIL import Image
import io

from utils.db_utils import (
    create_tables, SessionLocal, Student, Photo, 
    AttendanceSession, AttendanceRecord
)
from utils.face_utils import (
    extract_face_encoding, serialize_encoding, deserialize_encoding,
    detect_faces_in_frame, compare_faces, draw_face_boxes
)

# Ensure tables exist
create_tables()

# Create directories if they don't exist
os.makedirs("data/photos", exist_ok=True)

st.set_page_config(page_title="Multi-Face Attendance System", layout="wide")
st.title("🎓 Multi-Face Attendance System")

# Sidebar navigation
menu = ["Add Student", "View Students", "Take Attendance", "Attendance History", "Download CSV"]
choice = st.sidebar.selectbox("Navigation", menu)

# Helper functions
def save_uploaded_photo(uploaded_file, student_id):
    """Save uploaded photo and return file path."""
    if uploaded_file is not None:
        # Create student directory
        student_dir = f"data/photos/student_{student_id}"
        os.makedirs(student_dir, exist_ok=True)
        
        # Save file
        file_path = os.path.join(student_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    return None

def load_known_faces():
    """Load all known face encodings from database."""
    session = SessionLocal()
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
    
    session.close()
    return known_encodings, known_names, known_ids

# Page: Add Student
if choice == "Add Student":
    st.subheader("📝 Add New Student")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        with st.form("student_form"):
            name = st.text_input("Student Name")
            roll = st.text_input("Roll Number")
            uploaded_files = st.file_uploader(
                "Upload Photos (Multiple photos for better recognition)", 
                type=['jpg', 'jpeg', 'png'], 
                accept_multiple_files=True
            )
            submitted = st.form_submit_button("Add Student")
            
            if submitted and name and roll:
                session = SessionLocal()
                try:
                    # Check if roll number already exists
                    existing_student = session.query(Student).filter(Student.roll == roll).first()
                    if existing_student:
                        st.error("Roll number already exists!")
                    else:
                        # Add student
                        student = Student(name=name, roll=roll)
                        session.add(student)
                        session.commit()
                        
                        # Process and save photos
                        if uploaded_files:
                            for uploaded_file in uploaded_files:
                                # Save photo
                                file_path = save_uploaded_photo(uploaded_file, student.id)
                                
                                if file_path:
                                    # Extract face encoding
                                    encoding = extract_face_encoding(file_path)
                                    
                                    if encoding is not None:
                                        # Save photo record with encoding
                                        photo = Photo(
                                            student_id=student.id,
                                            photo_path=file_path,
                                            face_encoding=serialize_encoding(encoding)
                                        )
                                        session.add(photo)
                                    else:
                                        st.warning(f"No face detected in {uploaded_file.name}")
                            
                            session.commit()
                        
                        st.success(f"Student {name} added successfully!")
                        st.rerun()
                        
                except Exception as e:
                    session.rollback()
                    st.error(f"Error adding student: {e}")
                finally:
                    session.close()
            elif submitted:
                st.error("Please fill in all fields.")
    
    with col2:
        st.info("💡 Tips for better face recognition:")
        st.write("• Upload 2-3 clear photos per student")
        st.write("• Ensure good lighting")
        st.write("• Include different angles")
        st.write("• Avoid blurry images")

# Page: View Students
elif choice == "View Students":
    st.subheader("👥 Student List")
    
    session = SessionLocal()
    students = session.query(Student).all()
    
    if students:
        for student in students:
            with st.expander(f"{student.name} ({student.roll})"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**ID:** {student.id}")
                    st.write(f"**Added:** {student.created_at.strftime('%Y-%m-%d %H:%M')}")
                    
                    # Show photos
                    photos = session.query(Photo).filter(Photo.student_id == student.id).all()
                    st.write(f"**Photos:** {len(photos)}")
                    
                    if photos:
                        photo_cols = st.columns(min(len(photos), 3))
                        for i, photo in enumerate(photos[:3]):
                            if os.path.exists(photo.photo_path):
                                img = Image.open(photo.photo_path)
                                photo_cols[i].image(img, width=100)
                
                with col2:
                    if st.button(f"Delete {student.name}", key=f"del_{student.id}"):
                        # Delete student and associated photos
                        for photo in student.photos:
                            if os.path.exists(photo.photo_path):
                                os.remove(photo.photo_path)
                        session.delete(student)
                        session.commit()
                        st.success(f"Student {student.name} deleted!")
                        st.rerun()
    else:
        st.info("No students found. Add some students first!")
    
    session.close()

# Page: Take Attendance
elif choice == "Take Attendance":
    st.subheader("📷 Take Attendance")
    
    # Load known faces
    known_encodings, known_names, known_ids = load_known_faces()
    
    if not known_encodings:
        st.warning("No student photos found! Please add students with photos first.")
    else:
        st.info(f"Loaded {len(known_encodings)} known faces")
        
        # Session name input
        session_name = st.text_input("Session Name", value=f"Attendance_{datetime.now().strftime('%Y%m%d_%H%M')}")
        
        # Camera input
        camera_input = st.camera_input("Take a photo for attendance")
        
        if camera_input is not None:
            # Process the image
            image = Image.open(camera_input)
            image_array = np.array(image)
            if image_array is None or image_array.size == 0:
                st.error("Invalid image data")
                continue
            
            # Detect faces
            face_encodings, face_locations = detect_faces_in_frame(image_array)
            
            if face_encodings:
                st.success(f"Detected {len(face_encodings)} faces!")
                
                # Recognize faces
                recognized_students = []
                confidences = []
                names = []
                
                for face_encoding in face_encodings:
                    best_match_index = None
                    best_confidence = 0
                    
                    for i, known_encoding in enumerate(known_encodings):
                        is_match, confidence, _ = compare_faces([known_encoding], face_encoding)
                        if is_match and confidence > best_confidence:
                            best_confidence = confidence
                            best_match_index = i
                    
                    if best_match_index is not None and best_confidence > 0.6:
                        recognized_students.append(known_ids[best_match_index])
                        names.append(known_names[best_match_index])
                        confidences.append(best_confidence)
                    else:
                        names.append("Unknown")
                        confidences.append(0.0)
                
                # Draw face boxes
                annotated_image = draw_face_boxes(image_array.copy(), face_locations, names, confidences)
                st.image(annotated_image, caption="Detected Faces")
                
                # Show recognized students
                if recognized_students:
                    st.subheader("Recognized Students:")
                    recognized_df = pd.DataFrame({
                        'Name': [names[i] for i in range(len(names)) if names[i] != "Unknown"],
                        'Confidence': [f"{confidences[i]:.2f}" for i in range(len(confidences)) if names[i] != "Unknown"]
                    })
                    st.dataframe(recognized_df)
                
                # Save attendance button
                if st.button("Save Attendance") and session_name:
                    session = SessionLocal()
                    try:
                        # Create attendance session
                        attendance_session = AttendanceSession(session_name=session_name)
                        session.add(attendance_session)
                        session.commit()
                        
                        # Get all students
                        all_students = session.query(Student).all()
                        
                        # Mark attendance
                        for student in all_students:
                            if student.id in recognized_students:
                                # Present
                                idx = recognized_students.index(student.id)
                                confidence = float(confidences[idx]) if idx < len(confidences) else 0.0 if idx < len(confidences) else 0.0
                                record = AttendanceRecord(
                                    session_id=attendance_session.id,
                                    student_id=student.id,
                                    status='present',
                                    confidence=float(confidence)
                                )
                            else:
                                # Absent
                                record = AttendanceRecord(
                                    session_id=attendance_session.id,
                                    student_id=student.id,
                                    status='absent'
                                )
                            session.add(record)
                        
                        session.commit()
                        st.success(f"Attendance saved for session: {session_name}")
                        
                    except Exception as e:
                        session.rollback()
                        st.error(f"Error saving attendance: {e}")
                    finally:
                        session.close()
            else:
                st.warning("No faces detected in the image. Please try again.")

# Page: Attendance History
elif choice == "Attendance History":
    st.subheader("📊 Attendance History")
    
    session = SessionLocal()
    attendance_sessions = session.query(AttendanceSession).order_by(AttendanceSession.date.desc()).all()
    
    if attendance_sessions:
        for att_session in attendance_sessions:
            with st.expander(f"📅 {att_session.session_name} - {att_session.date.strftime('%Y-%m-%d %H:%M')}"):
                records = session.query(AttendanceRecord).filter(
                    AttendanceRecord.session_id == att_session.id
                ).join(Student).all()
                
                if records:
                    # Create dataframe
                    data = []
                    for record in records:
                        data.append({
                            'Name': record.student.name,
                            'Roll': record.student.roll,
                            'Status': record.status.title(),
                            'Confidence': record.confidence if record.confidence else 'N/A'
                        })
                    
                    df = pd.DataFrame(data)
                    st.dataframe(df, use_container_width=True)
                    
                    # Summary
                    present_count = len([r for r in records if r.status == 'present'])
                    total_count = len(records)
                    st.metric("Attendance Rate", f"{present_count}/{total_count} ({present_count/total_count*100:.1f}%)")
    else:
        st.info("No attendance records found.")
    
    session.close()

# Page: Download CSV
elif choice == "Download CSV":
    st.subheader("📥 Download Attendance CSV")
    
    session = SessionLocal()
    attendance_sessions = session.query(AttendanceSession).order_by(AttendanceSession.date.desc()).all()
    
    if attendance_sessions:
        # Select session
        session_options = {f"{s.session_name} - {s.date.strftime('%Y-%m-%d %H:%M')}": s.id for s in attendance_sessions}
        selected_session = st.selectbox("Select Session", options=list(session_options.keys()))
        
        if selected_session:
            session_id = session_options[selected_session]
            
            # Get attendance records
            records = session.query(AttendanceRecord).filter(
                AttendanceRecord.session_id == session_id
            ).join(Student).all()
            
            if records:
                # Create dataframe
                data = []
                for record in records:
                    data.append({
                        'Student_Name': record.student.name,
                        'Roll_Number': record.student.roll,
                        'Status': record.status.title(),
                        'Confidence': record.confidence if record.confidence else 'N/A',
                        'Date': attendance_sessions[0].date.strftime('%Y-%m-%d'),
                        'Time': attendance_sessions[0].date.strftime('%H:%M:%S')
                    })
                
                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True)
                
                # Download button
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"attendance_{selected_session.replace(' - ', '_').replace(':', '-')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No records found for this session.")
    else:
        st.info("No attendance sessions found.")
    
    session.close()

# Footer
st.markdown("---")
st.markdown("🎓 **Multi-Face Attendance System** - Built with Streamlit & Face Recognition")
