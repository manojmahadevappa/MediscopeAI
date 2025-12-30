import os
import logging
from fastapi import FastAPI

firebase_admin = None
firebase_app = None
firestore_client = None

groq_client = None

def init_firebase():
    global firebase_admin, firebase_app, firestore_client
    try:
        import firebase_admin as _firebase_admin
        from firebase_admin import credentials, firestore
        firebase_admin = _firebase_admin
        
        # Check if already initialized
        try:
            existing_app = firebase_admin.get_app()
            firestore_client = firestore.client()
            print("✅ Firebase already initialized (reusing existing app)")
            return
        except ValueError:
            # Not initialized yet, proceed
            pass
        
        # Check for credentials in multiple locations
        cred_path = os.getenv('FIREBASE_CREDENTIALS')
        
        if not cred_path or not os.path.exists(cred_path):
            # Try to find credentials file in webapp directory and project root
            webapp_dir = os.path.dirname(__file__)
            project_root = os.path.dirname(webapp_dir)
            
            possible_files = [
                os.path.join(webapp_dir, 'firebase-service-account.json'),
                os.path.join(webapp_dir, 'medioai-firebase-adminsdk-fbsvc-29aa5384ee.json'),
                os.path.join(project_root, 'medioai-firebase-adminsdk-fbsvc-29aa5384ee.json'),
                os.path.join(project_root, 'firebase-service-account.json'),
            ]
            
            for filepath in possible_files:
                if os.path.exists(filepath):
                    cred_path = filepath
                    break
        
        if cred_path and os.path.exists(cred_path):
            cred = credentials.Certificate(cred_path)
            firebase_app = firebase_admin.initialize_app(cred)
            firestore_client = firestore.client()
            print(f"✅ Firebase initialized successfully")
            print(f"   Using: {cred_path}")
        else:
            print("⚠️ FIREBASE_CREDENTIALS not set or file missing. Firebase will be disabled.")
            print(f"   Searched locations:")
            if not cred_path or not os.path.exists(cred_path):
                webapp_dir = os.path.dirname(__file__)
                project_root = os.path.dirname(webapp_dir)
                print(f"   - {os.path.join(webapp_dir, 'firebase-service-account.json')}")
                print(f"   - {os.path.join(project_root, 'medioai-firebase-adminsdk-fbsvc-29aa5384ee.json')}")        
    except Exception as e:
        print(f" Firebase initialization failed: {e}")
        firebase_admin = None
        firebase_app = None
        firestore_client = None

def init_groq():
    global groq_client
    try:
        from groq import Groq
        groq_key = os.getenv('GROQ_API_KEY')
        if groq_key:
            groq_client = Groq(api_key=groq_key)
            print(" Groq client initialized.")
        else:
            print(" GROQ_API_KEY not set. Groq support will be disabled.")
    except Exception as e:
        print(f" Groq initialization failed: {e}")
        groq_client = None

def register_startup(app: FastAPI):
    @app.on_event("startup")
    def _startup():
        init_firebase()
        init_groq()
