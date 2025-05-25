import streamlit as st
import bcrypt
import json
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import AUTH_CONFIG, DEFAULT_ADMIN

class AuthManager:
    def __init__(self):
        self.users_file = "data/users.json"
        self.sessions_file = "data/sessions.json"
        self.ensure_files_exist()
        self.create_default_admin()
    
    def ensure_files_exist(self):
        """Ensure user and session files exist."""
        os.makedirs("data", exist_ok=True)
        
        if not os.path.exists(self.users_file):
            with open(self.users_file, 'w') as f:
                json.dump({}, f)
        
        if not os.path.exists(self.sessions_file):
            with open(self.sessions_file, 'w') as f:
                json.dump({}, f)
    
    def create_default_admin(self):
        """Create default admin user if no users exist."""
        users = self.load_users()
        if not users:
            hashed_password = self.hash_password(DEFAULT_ADMIN['password'])
            users[DEFAULT_ADMIN['username']] = {
                'password': hashed_password,
                'email': DEFAULT_ADMIN['email'],
                'role': DEFAULT_ADMIN['role'],
                'created_at': datetime.now().isoformat(),
                'last_login': None,
                'login_attempts': 0,
                'is_active': True
            }
            self.save_users(users)
    
    def load_users(self) -> Dict[str, Any]:
        """Load users from file."""
        try:
            with open(self.users_file, 'r') as f:
                return json.load(f)
        except:
            return {}
    
    def save_users(self, users: Dict[str, Any]):
        """Save users to file."""
        with open(self.users_file, 'w') as f:
            json.dump(users, f, indent=2)
    
    def load_sessions(self) -> Dict[str, Any]:
        """Load sessions from file."""
        try:
            with open(self.sessions_file, 'r') as f:
                return json.load(f)
        except:
            return {}
    
    def save_sessions(self, sessions: Dict[str, Any]):
        """Save sessions to file."""
        with open(self.sessions_file, 'w') as f:
            json.dump(sessions, f, indent=2)
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt."""
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash."""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    
    def create_user(self, username: str, password: str, email: str, role: str = 'teacher') -> bool:
        """Create a new user."""
        users = self.load_users()
        
        if username in users:
            return False
        
        users[username] = {
            'password': self.hash_password(password),
            'email': email,
            'role': role,
            'created_at': datetime.now().isoformat(),
            'last_login': None,
            'login_attempts': 0,
            'is_active': True
        }
        
        self.save_users(users)
        return True
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user login."""
        users = self.load_users()
        
        if username not in users:
            return None
        
        user = users[username]
        
        # Check if user is active
        if not user.get('is_active', True):
            return None
        
        # Check login attempts
        if user.get('login_attempts', 0) >= AUTH_CONFIG['max_login_attempts']:
            return None
        
        # Verify password
        if self.verify_password(password, user['password']):
            # Reset login attempts and update last login
            user['login_attempts'] = 0
            user['last_login'] = datetime.now().isoformat()
            users[username] = user
            self.save_users(users)
            
            # Create session
            session_id = self.create_session(username)
            user['session_id'] = session_id
            
            return user
        else:
            # Increment login attempts
            user['login_attempts'] = user.get('login_attempts', 0) + 1
            users[username] = user
            self.save_users(users)
            return None
    
    def create_session(self, username: str) -> str:
        """Create a new session for user."""
        import uuid
        session_id = str(uuid.uuid4())
        
        sessions = self.load_sessions()
        sessions[session_id] = {
            'username': username,
            'created_at': datetime.now().isoformat(),
            'expires_at': (datetime.now() + timedelta(hours=AUTH_CONFIG['session_timeout_hours'])).isoformat()
        }
        
        self.save_sessions(sessions)
        return session_id
    
    def validate_session(self, session_id: str) -> Optional[str]:
        """Validate session and return username if valid."""
        sessions = self.load_sessions()
        
        if session_id not in sessions:
            return None
        
        session = sessions[session_id]
        expires_at = datetime.fromisoformat(session['expires_at'])
        
        if datetime.now() > expires_at:
            # Session expired, remove it
            del sessions[session_id]
            self.save_sessions(sessions)
            return None
        
        return session['username']
    
    def logout_user(self, session_id: str):
        """Logout user by removing session."""
        sessions = self.load_sessions()
        if session_id in sessions:
            del sessions[session_id]
            self.save_sessions(sessions)
    
    def get_user_info(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user information."""
        users = self.load_users()
        return users.get(username)
    
    def update_user(self, username: str, updates: Dict[str, Any]) -> bool:
        """Update user information."""
        users = self.load_users()
        
        if username not in users:
            return False
        
        users[username].update(updates)
        self.save_users(users)
        return True
    
    def delete_user(self, username: str) -> bool:
        """Delete user."""
        users = self.load_users()
        
        if username not in users:
            return False
        
        del users[username]
        self.save_users(users)
        return True
    
    def get_all_users(self) -> Dict[str, Any]:
        """Get all users."""
        return self.load_users()

# Streamlit authentication decorators and helpers
def require_auth(roles=None):
    """Decorator to require authentication for a function."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not is_authenticated():
                show_login_page()
                return None
            
            if roles and not has_role(roles):
                st.error("Access denied. Insufficient permissions.")
                return None
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

def is_authenticated() -> bool:
    """Check if user is authenticated."""
    return 'session_id' in st.session_state and 'username' in st.session_state

def has_role(required_roles) -> bool:
    """Check if user has required role."""
    if not is_authenticated():
        return False
    
    user_role = st.session_state.get('user_role', '')
    if isinstance(required_roles, str):
        required_roles = [required_roles]
    
    return user_role in required_roles

def get_current_user() -> Optional[str]:
    """Get current authenticated username."""
    return st.session_state.get('username')

def show_login_page():
    """Display login page."""
    st.title("🔐 Login - Multi-Face Attendance System")
    
    auth_manager = AuthManager()
    
    with st.form("login_form"):
        st.subheader("Please log in to continue")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_button = st.form_submit_button("Login")
        
        if login_button:
            if username and password:
                user = auth_manager.authenticate_user(username, password)
                if user:
                    st.session_state['session_id'] = user['session_id']
                    st.session_state['username'] = username
                    st.session_state['user_role'] = user['role']
                    st.session_state['user_email'] = user['email']
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid credentials or account locked.")
            else:
                st.error("Please enter both username and password.")
    
    with st.expander("Default Admin Credentials (Change after first login)"):
        st.warning("**Default Admin:**")
        st.code(f"Username: {DEFAULT_ADMIN['username']}")
        st.code(f"Password: {DEFAULT_ADMIN['password']}")

def logout():
    """Logout current user."""
    if 'session_id' in st.session_state:
        auth_manager = AuthManager()
        auth_manager.logout_user(st.session_state['session_id'])
    
    # Clear session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    st.rerun() 