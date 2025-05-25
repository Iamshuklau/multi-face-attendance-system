import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EMAIL_CONFIG, NOTIFICATION_CONFIG
from utils.db_utils import SessionLocal, Notification, Student, log_audit_event

logger = logging.getLogger(__name__)

class NotificationManager:
    """Manages notifications and alerts - Simplified version for Python 3.13 compatibility."""
    
    def __init__(self):
        self.enabled = EMAIL_CONFIG.get('enabled', False)
    
    def send_email(self, to_email: str, subject: str, body: str, attachments: List[str] = None) -> bool:
        """Log email notification (email sending disabled for Python 3.13 compatibility)."""
        try:
            logger.info(f"📧 Email notification: {subject} -> {to_email}")
            logger.info(f"Body: {body[:100]}...")
            print(f"📧 Email notification logged: {subject} -> {to_email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to log email to {to_email}: {e}")
            return False
    
    def send_attendance_alert(self, student_id: int, attendance_percentage: float, session_name: str):
        """Send low attendance alert notification."""
        session = SessionLocal()
        try:
            student = session.query(Student).filter(Student.id == student_id).first()
            if not student:
                logger.error(f"Student {student_id} not found")
                return
            
            # Create notification record
            notification = Notification(
                student_id=student_id,
                type='low_attendance',
                title='Low Attendance Alert',
                message=f'Attendance for {session_name} is {attendance_percentage:.1f}%',
                priority='high'
            )
            session.add(notification)
            
            # Log notification
            subject = f"Low Attendance Alert - {student.name}"
            body = f"Student {student.name} has low attendance: {attendance_percentage:.1f}%"
            
            if student.email and NOTIFICATION_CONFIG['email_notifications']:
                if self.send_email(student.email, subject, body):
                    notification.sent_to_student = True
            
            if student.parent_email and NOTIFICATION_CONFIG['email_notifications']:
                if self.send_email(student.parent_email, subject, body):
                    notification.sent_to_parent = True
            
            notification.sent_at = datetime.utcnow()
            session.commit()
            
            # Log audit event
            log_audit_event(
                session, 
                'system', 
                'send_attendance_alert',
                'notification',
                notification.id,
                f"Low attendance alert created for student {student.name}"
            )
            
        except Exception as e:
            logger.error(f"Failed to send attendance alert: {e}")
            session.rollback()
        finally:
            session.close()
    
    def send_absence_notification(self, student_id: int, session_name: str, date: str):
        """Send absence notification."""
        session = SessionLocal()
        try:
            student = session.query(Student).filter(Student.id == student_id).first()
            if not student or not student.parent_email:
                return
            
            notification = Notification(
                student_id=student_id,
                type='absence_alert',
                title='Student Absence Notification',
                message=f'Student was absent from {session_name} on {date}',
                priority='normal'
            )
            session.add(notification)
            
            subject = f"Absence Notification - {student.name}"
            body = f"Student {student.name} was absent from {session_name} on {date}"
            
            if self.send_email(student.parent_email, subject, body):
                notification.sent_to_parent = True
                notification.sent_at = datetime.utcnow()
            
            session.commit()
            
        except Exception as e:
            logger.error(f"Failed to send absence notification: {e}")
            session.rollback()
        finally:
            session.close()
    
    def send_system_alert(self, email: str, alert_type: str, message: str):
        """Send system alerts."""
        try:
            subject = f"System Alert - {alert_type}"
            body = f"Alert: {alert_type} - {message} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            return self.send_email(email, subject, body)
        except Exception as e:
            logger.error(f"Failed to send system alert: {e}")
            return False

def get_unread_notifications(student_id: int = None, limit: int = 50):
    """Get unread notifications."""
    session = SessionLocal()
    try:
        query = session.query(Notification).filter(Notification.is_read == False)
        
        if student_id:
            query = query.filter(Notification.student_id == student_id)
        
        notifications = query.order_by(Notification.created_at.desc()).limit(limit).all()
        return notifications
        
    except Exception as e:
        logger.error(f"Error getting notifications: {e}")
        return []
    finally:
        session.close()

def mark_notification_as_read(notification_id: int):
    """Mark a notification as read."""
    session = SessionLocal()
    try:
        notification = session.query(Notification).filter(
            Notification.id == notification_id
        ).first()
        
        if notification:
            notification.is_read = True
            session.commit()
            return True
        
    except Exception as e:
        logger.error(f"Error marking notification as read: {e}")
        session.rollback()
    finally:
        session.close()
    
    return False
