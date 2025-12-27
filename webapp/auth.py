"""
Authentication utilities for MediscopeAI
Handles password hashing, token generation, and session management
"""
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import Optional
import secrets
import hashlib

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# In-memory session store (use Redis for production)
active_sessions = {}

# Session configuration
SESSION_DURATION = timedelta(hours=24)


def hash_password(password: str) -> str:
    """Hash a password for storage"""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)


def create_session_token(user_id: int, username: str) -> str:
    """Create a new session token for user"""
    token = secrets.token_urlsafe(32)
    active_sessions[token] = {
        'user_id': user_id,
        'username': username,
        'created_at': datetime.utcnow(),
        'expires_at': datetime.utcnow() + SESSION_DURATION
    }
    return token


def get_current_user(token: str) -> Optional[dict]:
    """Get user from session token"""
    session = active_sessions.get(token)
    if not session:
        return None
    
    # Check expiration
    if datetime.utcnow() > session['expires_at']:
        del active_sessions[token]
        return None
    
    return session


def invalidate_session(token: str):
    """Remove session token (logout)"""
    if token in active_sessions:
        del active_sessions[token]


def cleanup_expired_sessions():
    """Remove expired sessions"""
    now = datetime.utcnow()
    expired = [token for token, session in active_sessions.items() 
               if now > session['expires_at']]
    for token in expired:
        del active_sessions[token]
