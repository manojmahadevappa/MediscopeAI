"""
Firebase configuration and initialization for MediscopeAI
"""
import os
import json
import firebase_admin
from firebase_admin import credentials, firestore, auth
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

# Initialize Firebase Admin SDK
def init_firebase():
    """Initialize Firebase Admin SDK with service account"""
    try:
        # Check if already initialized
        firebase_admin.get_app()
        print("✅ Firebase already initialized")
        return True
    except ValueError:
        # Not initialized, proceed with initialization
        pass
    
    try:
        # Try to load from environment variable first
        firebase_creds = os.getenv('FIREBASE_CREDENTIALS')
        
        if firebase_creds:
            # Parse JSON from environment variable
            cred_dict = json.loads(firebase_creds)
            cred = credentials.Certificate(cred_dict)
        else:
            # Fall back to service account file
            cred_path = os.path.join(os.path.dirname(__file__), 'firebase-service-account.json')
            if os.path.exists(cred_path):
                cred = credentials.Certificate(cred_path)
            else:
                print("⚠️ Firebase credentials not found. Please set FIREBASE_CREDENTIALS env var or create firebase-service-account.json")
                print("   Get your credentials from: https://console.firebase.google.com/project/_/settings/serviceaccounts/adminsdk")
                return False
        
        firebase_admin.initialize_app(cred)
        print("✅ Firebase initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Firebase initialization failed: {e}")
        return False


# Get Firestore client
def get_firestore_db():
    """Get Firestore database client"""
    return firestore.client()


class FirebaseAuth:
    """Firebase Authentication helper class"""
    
    @staticmethod
    def create_user(email: str, password: str, username: str) -> Dict[str, Any]:
        """Create a new user in Firebase Auth"""
        try:
            # Create user in Firebase Auth
            user = auth.create_user(
                email=email,
                password=password,
                display_name=username
            )
            
            # Store additional user data in Firestore
            db = get_firestore_db()
            db.collection('users').document(user.uid).set({
                'username': username,
                'email': email,
                'created_at': firestore.SERVER_TIMESTAMP,
                'total_analyses': 0,
                'total_messages': 0
            })
            
            return {
                'success': True,
                'user_id': user.uid,
                'message': 'User created successfully'
            }
            
        except auth.EmailAlreadyExistsError:
            return {'success': False, 'error': 'Email already exists'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    
    @staticmethod
    def verify_user(email: str, password: str) -> Dict[str, Any]:
        """Verify user credentials (using Firebase REST API)"""
        import requests
        
        try:
            # Firebase Web API key from environment or default
            api_key = os.getenv('FIREBASE_WEB_API_KEY', 'AIzaSyAVZxSPlhtKN-28R1VHWoVjzkhKqKTe0O8')
            
            # Call Firebase Auth REST API
            url = f'https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={api_key}'
            payload = {
                'email': email,
                'password': password,
                'returnSecureToken': True
            }
            
            response = requests.post(url, json=payload)
            data = response.json()
            
            if response.status_code == 200:
                # Get user data from Firestore
                db = get_firestore_db()
                user_doc = db.collection('users').document(data['localId']).get()
                user_data = user_doc.to_dict() if user_doc.exists else {}
                
                return {
                    'success': True,
                    'user_id': data['localId'],
                    'token': data['idToken'],
                    'refresh_token': data['refreshToken'],
                    'username': user_data.get('username', email.split('@')[0])
                }
            else:
                return {
                    'success': False,
                    'error': data.get('error', {}).get('message', 'Invalid credentials')
                }
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    
    @staticmethod
    def verify_token(token: str) -> Optional[Dict[str, Any]]:
        """Verify Firebase ID token"""
        try:
            decoded_token = auth.verify_id_token(token)
            
            # Get user data from Firestore
            db = get_firestore_db()
            user_doc = db.collection('users').document(decoded_token['uid']).get()
            user_data = user_doc.to_dict() if user_doc.exists else {}
            
            return {
                'user_id': decoded_token['uid'],
                'email': decoded_token.get('email'),
                'username': user_data.get('username', decoded_token.get('name', 'User'))
            }
            
        except auth.InvalidIdTokenError:
            return None
        except auth.ExpiredIdTokenError:
            return None
        except Exception as e:
            print(f"Token verification error: {e}")
            return None
    
    
    @staticmethod
    def delete_user(user_id: str) -> bool:
        """Delete user from Firebase Auth and Firestore"""
        try:
            auth.delete_user(user_id)
            db = get_firestore_db()
            db.collection('users').document(user_id).delete()
            return True
        except Exception as e:
            print(f"Delete user error: {e}")
            return False


class FirebaseAnalysis:
    """Firebase Analysis data management"""
    
    @staticmethod
    def save_analysis(user_id: str, analysis_data: Dict[str, Any]) -> str:
        """Save analysis to Firestore"""
        try:
            db = get_firestore_db()
            
            # Add timestamp and user_id
            analysis_data['user_id'] = user_id
            analysis_data['created_at'] = firestore.SERVER_TIMESTAMP
            
            # Save analysis
            doc_ref = db.collection('analyses').document()
            doc_ref.set(analysis_data)
            
            # Update user stats
            user_ref = db.collection('users').document(user_id)
            user_ref.update({
                'total_analyses': firestore.Increment(1)
            })
            
            return doc_ref.id
            
        except Exception as e:
            print(f"Save analysis error: {e}")
            raise
    
    
    @staticmethod
    def get_user_analyses(user_id: str, limit: int = 50) -> list:
        """Get all analyses for a user"""
        try:
            db = get_firestore_db()
            # Simplified query without order_by to avoid index requirement
            analyses = db.collection('analyses')\
                .where('user_id', '==', user_id)\
                .limit(limit)\
                .stream()
            
            results = []
            for doc in analyses:
                data = doc.to_dict()
                data['id'] = doc.id
                # Convert timestamp to ISO string
                if 'created_at' in data and data['created_at']:
                    try:
                        # Handle Firestore timestamp
                        if hasattr(data['created_at'], 'isoformat'):
                            data['created_at'] = data['created_at'].isoformat()
                        else:
                            data['created_at'] = str(data['created_at'])
                    except:
                        data['created_at'] = datetime.utcnow().isoformat()
                else:
                    data['created_at'] = datetime.utcnow().isoformat()
                results.append(data)
            
            # Sort by created_at in Python (newest first)
            results.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            
            return results[:limit]
            
        except Exception as e:
            print(f"Get analyses error: {e}")
            return []
    
    
    @staticmethod
    def get_analysis(analysis_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific analysis by ID"""
        try:
            db = get_firestore_db()
            doc = db.collection('analyses').document(analysis_id).get()
            
            if doc.exists:
                data = doc.to_dict()
                data['id'] = doc.id
                if 'created_at' in data and data['created_at']:
                    data['created_at'] = data['created_at'].isoformat()
                return data
            
            return None
            
        except Exception as e:
            print(f"Get analysis error: {e}")
            return None


class FirebaseChat:
    """Firebase Chat message management"""
    
    @staticmethod
    def save_message(analysis_id: str, role: str, content: str) -> str:
        """Save chat message to Firestore"""
        try:
            db = get_firestore_db()
            
            message_data = {
                'analysis_id': analysis_id,
                'role': role,
                'content': content,
                'created_at': firestore.SERVER_TIMESTAMP
            }
            
            doc_ref = db.collection('chat_messages').document()
            doc_ref.set(message_data)
            
            # Update analysis chat count
            analysis_ref = db.collection('analyses').document(analysis_id)
            analysis_ref.update({
                'chat_count': firestore.Increment(1)
            })
            
            return doc_ref.id
            
        except Exception as e:
            print(f"Save message error: {e}")
            raise
    
    
    @staticmethod
    def get_analysis_messages(analysis_id: str) -> list:
        """Get all chat messages for an analysis"""
        try:
            db = get_firestore_db()
            messages = db.collection('chat_messages')\
                .where('analysis_id', '==', analysis_id)\
                .order_by('created_at')\
                .stream()
            
            results = []
            for doc in messages:
                data = doc.to_dict()
                data['id'] = doc.id
                if 'created_at' in data and data['created_at']:
                    data['created_at'] = data['created_at'].isoformat()
                results.append(data)
            
            return results
            
        except Exception as e:
            print(f"Get messages error: {e}")
            return []
    
    
    @staticmethod
    def get_user_total_messages(user_id: str) -> int:
        """Get total message count for a user"""
        try:
            db = get_firestore_db()
            
            # Get all analyses for user
            analyses = db.collection('analyses')\
                .where('user_id', '==', user_id)\
                .stream()
            
            total = 0
            for analysis in analyses:
                data = analysis.to_dict()
                total += data.get('chat_count', 0)
            
            return total
            
        except Exception as e:
            print(f"Get total messages error: {e}")
            return 0
