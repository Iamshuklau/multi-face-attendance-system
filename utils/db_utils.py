from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, DateTime, Boolean, LargeBinary, Text, Float
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime
import sys
import os

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATABASE_URL

engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=300)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class AcademicYear(Base):
    __tablename__ = 'academic_years'
    id = Column(Integer, primary_key=True, index=True)
    year_name = Column(String, nullable=False)  # e.g., "2023-2024"
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    classes = relationship("Class", back_populates="academic_year")
    semesters = relationship("Semester", back_populates="academic_year")

class Semester(Base):
    __tablename__ = 'semesters'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)  # e.g., "Fall 2023", "Spring 2024"
    academic_year_id = Column(Integer, ForeignKey('academic_years.id'))
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    academic_year = relationship("AcademicYear", back_populates="semesters")
    classes = relationship("Class", back_populates="semester")

class Class(Base):
    __tablename__ = 'classes'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)  # e.g., "Grade 10-A", "Computer Science 101"
    code = Column(String, unique=True, nullable=False)  # e.g., "CS101", "GR10A"
    description = Column(Text)
    academic_year_id = Column(Integer, ForeignKey('academic_years.id'))
    semester_id = Column(Integer, ForeignKey('semesters.id'))
    teacher_username = Column(String, nullable=False)  # Reference to auth system
    max_students = Column(Integer, default=100)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    academic_year = relationship("AcademicYear", back_populates="classes")
    semester = relationship("Semester", back_populates="classes")
    students = relationship("Student", back_populates="class_obj")
    subjects = relationship("Subject", back_populates="class_obj")
    attendance_sessions = relationship("AttendanceSession", back_populates="class_obj")

class Subject(Base):
    __tablename__ = 'subjects'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    code = Column(String, nullable=False)
    class_id = Column(Integer, ForeignKey('classes.id'))
    teacher_username = Column(String, nullable=False)
    credits = Column(Integer, default=3)
    description = Column(Text)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    class_obj = relationship("Class", back_populates="subjects")
    attendance_sessions = relationship("AttendanceSession", back_populates="subject")

class Student(Base):
    __tablename__ = 'students'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    roll = Column(String, nullable=False)
    email = Column(String)
    phone = Column(String)
    parent_email = Column(String)
    parent_phone = Column(String)
    class_id = Column(Integer, ForeignKey('classes.id'))
    admission_date = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    class_obj = relationship("Class", back_populates="students")
    photos = relationship("Photo", back_populates="student", cascade="all, delete-orphan")
    attendance_records = relationship("AttendanceRecord", back_populates="student")
    notifications = relationship("Notification", back_populates="student")

class Photo(Base):
    __tablename__ = 'photos'
    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey('students.id', ondelete='CASCADE'))
    photo_path = Column(String, nullable=False)
    face_encoding = Column(LargeBinary)  # Store face embeddings as binary
    is_primary = Column(Boolean, default=False)  # Mark primary photo
    quality_score = Column(Float)  # Photo quality score
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship to student
    student = relationship("Student", back_populates="photos")

class AttendanceSession(Base):
    __tablename__ = 'attendance_sessions'
    id = Column(Integer, primary_key=True, index=True)
    session_name = Column(String, nullable=False)
    class_id = Column(Integer, ForeignKey('classes.id'))
    subject_id = Column(Integer, ForeignKey('subjects.id'))
    session_type = Column(String, default='regular')  # regular, exam, lab, etc.
    date = Column(DateTime, default=datetime.utcnow)
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    location = Column(String)
    notes = Column(Text)
    total_present = Column(Integer, default=0)
    total_absent = Column(Integer, default=0)
    total_late = Column(Integer, default=0)
    conducted_by = Column(String, nullable=False)  # Teacher username
    is_finalized = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    class_obj = relationship("Class", back_populates="attendance_sessions")
    subject = relationship("Subject", back_populates="attendance_sessions")
    records = relationship("AttendanceRecord", back_populates="session", cascade="all, delete-orphan")

class AttendanceRecord(Base):
    __tablename__ = 'attendance_records'
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey('attendance_sessions.id', ondelete='CASCADE'))
    student_id = Column(Integer, ForeignKey('students.id'))
    status = Column(String, default='present')  # present, absent, late, excused
    confidence = Column(Float)  # Face recognition confidence
    marked_at = Column(DateTime, default=datetime.utcnow)
    marked_by = Column(String)  # auto, manual_teacher, manual_admin
    late_minutes = Column(Integer, default=0)
    notes = Column(Text)
    
    # Relationships
    session = relationship("AttendanceSession", back_populates="records")
    student = relationship("Student", back_populates="attendance_records")

class Notification(Base):
    __tablename__ = 'notifications'
    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey('students.id'))
    type = Column(String, nullable=False)  # attendance_alert, low_attendance, etc.
    title = Column(String, nullable=False)
    message = Column(Text, nullable=False)
    sent_to_student = Column(Boolean, default=False)
    sent_to_parent = Column(Boolean, default=False)
    sent_at = Column(DateTime)
    is_read = Column(Boolean, default=False)
    priority = Column(String, default='normal')  # low, normal, high, urgent
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    student = relationship("Student", back_populates="notifications")

class AuditLog(Base):
    __tablename__ = 'audit_logs'
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, nullable=False)
    action = Column(String, nullable=False)  # login, logout, add_student, take_attendance, etc.
    resource_type = Column(String)  # student, attendance, user, etc.
    resource_id = Column(String)
    details = Column(Text)  # JSON string with additional details
    ip_address = Column(String)
    user_agent = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

class SystemSettings(Base):
    __tablename__ = 'system_settings'
    id = Column(Integer, primary_key=True, index=True)
    key = Column(String, unique=True, nullable=False)
    value = Column(Text, nullable=False)
    category = Column(String, default='general')
    description = Column(Text)
    updated_by = Column(String, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow)

class PerformanceMetrics(Base):
    __tablename__ = 'performance_metrics'
    id = Column(Integer, primary_key=True, index=True)
    metric_name = Column(String, nullable=False)
    metric_value = Column(Float, nullable=False)
    unit = Column(String)  # seconds, percentage, count, etc.
    category = Column(String)  # face_recognition, database, system, etc.
    details = Column(Text)  # JSON string with additional data
    recorded_at = Column(DateTime, default=datetime.utcnow)

def create_tables():
    """Create all tables."""
    Base.metadata.create_all(bind=engine)

def get_session():
    """Get database session."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

# Utility functions for common database operations
def get_active_academic_year(session):
    """Get the currently active academic year."""
    return session.query(AcademicYear).filter(AcademicYear.is_active == True).first()

def get_active_semester(session):
    """Get the currently active semester."""
    return session.query(Semester).filter(Semester.is_active == True).first()

def get_classes_for_teacher(session, username):
    """Get all classes assigned to a teacher."""
    return session.query(Class).filter(
        Class.teacher_username == username,
        Class.is_active == True
    ).all()

def get_students_in_class(session, class_id):
    """Get all active students in a class."""
    return session.query(Student).filter(
        Student.class_id == class_id,
        Student.is_active == True
    ).all()

def log_audit_event(session, username, action, resource_type=None, resource_id=None, details=None, ip_address=None, user_agent=None):
    """Log an audit event."""
    audit_log = AuditLog(
        username=username,
        action=action,
        resource_type=resource_type,
        resource_id=str(resource_id) if resource_id else None,
        details=details,
        ip_address=ip_address,
        user_agent=user_agent
    )
    session.add(audit_log)
    session.commit()

def record_performance_metric(session, metric_name, metric_value, unit=None, category=None, details=None):
    """Record a performance metric."""
    metric = PerformanceMetrics(
        metric_name=metric_name,
        metric_value=metric_value,
        unit=unit,
        category=category,
        details=details
    )
    session.add(metric)
    session.commit() 