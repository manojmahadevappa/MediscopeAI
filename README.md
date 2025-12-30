# 🧠 MediscopeAI - Brain Tumor Detection System

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Firebase](https://img.shields.io/badge/Firebase-Firestore-orange.svg)](https://firebase.google.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Research-yellow.svg)](LICENSE)

**MediscopeAI** is an advanced AI-powered medical diagnostic platform that leverages deep learning to analyze CT and MRI brain scans for tumor detection, classification, and prognostic analysis. The system provides instant, accurate diagnostics with AI explainability (Grad-CAM), survival rate predictions, and an integrated medical consultation chatbot.

---

## 🌟 Features

### Core Capabilities
- **🔬 Multi-Modal Analysis**: Supports both CT and MRI scan uploads (single or combined)
- **🎯 Advanced Classification**:
  - Multiclass Classification: Healthy, Benign, Malignant
  - Severity Assessment: Multiple severity levels for tumor staging
  - Single-stage inference for optimal performance
- **📊 AI Explainability**: Grad-CAM heatmaps showing model attention regions
- **🩺 Survival Rate Prediction**: AI-powered prognosis using Groq LLama 3.3-70b
- **💬 AI Medical Assistant**: Interactive chatbot for medical consultation
- **📱 User Dashboard**: Track analysis history, download reports
- **🔐 Firebase Authentication**: Secure user accounts and data storage
- **📄 Comprehensive Reports**: Downloadable PDF reports with full diagnostic details

### Technical Highlights
- **Deep Learning Models**: ResNet50 architecture with ImageNet transfer learning
- **Real-time Inference**: Sub-200ms predictions with model caching
- **Service Initialization**: Robust Firebase & Groq initialization with graceful degradation
- **Cloud Storage**: Firebase Firestore for scalable data management
- **Responsive UI**: Modern, medical-grade interface with Tailwind CSS
- **Smart Authentication**: Beautiful modal-based auth flow with friendly UX
- **HIPAA-Compliant Design**: Privacy-focused architecture
- **Production Ready**: Streamlined deployment with environment-based configuration

---

## 📋 Table of Contents

- [System Architecture](#-system-architecture)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Running the Application](#-running-the-application)
- [Usage Guide](#-usage-guide)
- [Project Structure](#-project-structure)
- [Model Training](#-model-training)
- [API Documentation](#-api-documentation)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🏗️ System Architecture

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│   Web Frontend  │─────▶│  FastAPI Backend │─────▶│  PyTorch Models │
│  (HTML/JS/CSS)  │      │   (Python 3.10)  │      │  (ResNet-based) │
└─────────────────┘      └──────────────────┘      └─────────────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
            ┌───────────┐  ┌──────────┐  ┌──────────┐
            │  Firebase │  │   Groq   │  │  Grad-   │
            │ Firestore │  │   API    │  │   CAM    │
            └───────────┘  └──────────┘  └──────────┘
```

**Components:**
- **Frontend**: Responsive HTML templates with vanilla JavaScript
- **Backend**: FastAPI with async support
- **Database**: Firebase Firestore (NoSQL)
- **Authentication**: Firebase Auth with JWT tokens
- **AI Models**: PyTorch models (multiclass, multimodal)
- **LLM Integration**: Groq API for medical consultation and prognosis
- **Explainability**: Grad-CAM implementation for visual interpretation

---

## ✅ Prerequisites

### System Requirements
- **Operating System**: Windows 10/11, macOS, or Linux
- **Python**: 3.10 or higher (3.11 recommended)
- **RAM**: Minimum 8GB (16GB recommended for model training)
- **Storage**: 5GB free space (for models and dataset)
- **GPU**: Optional (CUDA-compatible GPU for faster inference)

### Required Accounts
1. **Firebase Account** (free tier sufficient):
   - Create project at [Firebase Console](https://console.firebase.google.com/)
   - Enable Firestore Database
   - Enable Authentication (Email/Password)
   - Download service account JSON

2. **Groq API Account** (free tier available):
   - Sign up at [Groq Console](https://console.groq.com/)
   - Generate API key

---

## 🚀 Installation

### Step 1: Clone the Repository

```powershell
git clone https://github.com/manojmahadevappa/Brain-Tumer-AI.git
cd "Brain Tumer Project"
```

### Step 2: Create Virtual Environment

```powershell
# Windows PowerShell
python -m venv env
.\env\Scripts\Activate.ps1

# macOS/Linux
python3 -m venv env
source env/bin/activate
```

### Step 3: Install Dependencies

```powershell
# Upgrade pip first
python -m pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt
```

**Core Dependencies:**
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `torch` & `torchvision` - Deep learning
- `firebase-admin` - Firebase SDK
- `groq` - LLM API client
- `pillow` - Image processing
- `numpy` - Numerical computing
- `python-multipart` - File upload support

### Step 4: Download Pre-trained Models

The repository includes three pre-trained model files:
- `model_binary.pth` - Binary classification (Healthy vs Tumor)
- `model_multiclass.pth` - Three-class classification
- `model_basic.pth` - Basic model for initial testing

**Models are already included in the repository.** If missing, train new models following the [Model Training](#-model-training) section.

---

## ⚙️ Configuration

### 1. Firebase Setup

#### Create Firebase Project
1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Click "Add project" and follow the wizard
3. Enable Firestore Database:
   - Go to Firestore Database → Create database
   - Start in **production mode**
   - Choose your region

#### Configure Firestore Rules
```javascript
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    match /users/{userId} {
      allow read, write: if request.auth != null && request.auth.uid == userId;
    }
    match /analyses/{analysisId} {
      allow read, write: if request.auth != null;
    }
    match /chat_messages/{messageId} {
      allow read, write: if request.auth != null;
    }
  }
}
```

#### Enable Authentication
1. Go to Authentication → Sign-in method
2. Enable **Email/Password** provider

#### Download Service Account
1. Go to Project Settings → Service Accounts
2. Click "Generate new private key"
3. Save as `webapp/firebase-service-account.json`

**⚠️ IMPORTANT**: Keep this file secure and never commit to GitHub!

### 2. Environment Variables

Create `.env` file in project root:

```env
# Firebase Configuration
FIREBASE_PROJECT_ID=your-project-id
FIREBASE_PRIVATE_KEY_ID=your-private-key-id
FIREBASE_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
FIREBASE_CLIENT_EMAIL=your-service-account@your-project.iam.gserviceaccount.com

# Groq API Configuration
GROQ_API_KEY=gsk_your_groq_api_key_here

# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=True
```

### 3. Firebase Config File

Ensure `webapp/firebase-service-account.json` has this structure:

```json
{
  "type": "service_account",
  "project_id": "your-project-id",
  "private_key_id": "your-key-id",
  "private_key": "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n",
  "client_email": "firebase-adminsdk-xxxxx@your-project.iam.gserviceaccount.com",
  "client_id": "123456789",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/..."
}
```

---

## ▶️ Running the Application

### Start the Server

```powershell
# Activate virtual environment first
.\env\Scripts\Activate.ps1  # Windows
source env/bin/activate      # macOS/Linux

# Navigate to project root
cd "C:\Users\YourName\Documents\Brain Tumer Project"

# Start FastAPI server with auto-reload
python -m uvicorn webapp.app:app --host 0.0.0.0 --port 8000 --reload
```

### Expected Output

```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [12345] using StatReload
INFO:     Started server process [67890]
INFO:     Waiting for application startup.
✅ Firebase initialized successfully
✅ MediscopeAI server started with Firebase authentication
INFO:     Application startup complete.
```

### Access the Application

Open your browser and navigate to:
- **Homepage**: http://localhost:8000/
- **Login**: http://localhost:8000/login
- **Dashboard**: http://localhost:8000/dashboard

---

## 📖 Usage Guide

### 1. Create an Account

1. Navigate to http://localhost:8000/login
2. Click "Sign up here"
3. Enter email and password
4. Click "Create Account"

### 2. Analyze Brain Scans

1. **Login** to your account
2. Go to **Homepage** (http://localhost:8000/)
3. Fill in **Patient Information**:
   - Patient Name
   - Age (in years)
   - Gender (Male/Female/Other)
   - Clinical Notes (optional)
4. **Upload Scans**:
   - CT Scan (optional)
   - MRI Scan (optional)
   - At least one scan required
5. Click **"Analyze Scans with AI"**

### 3. Authentication

- **Public Pages**: Homepage (/) - accessible to all users
- **Protected Features**: Analysis (/predict), Results (/results), Chatbot (/chatbot), Assistant (/assistant)
- **Smart Redirect**: Attempting to access protected features without login shows a beautiful modal:
  - Elegant popup with authentication message
  - "Sign In to Continue" button
  - Auto-dismiss after 6 seconds
  - Smooth fade-in of login form after dismissal
- **Session Management**: JWT tokens stored in localStorage

### 4. View Results

After analysis completes (~2-5 seconds):
- **Diagnosis Card**: Shows prediction (Healthy/Benign/Malignant) with confidence
- **Patient Information**: Displays entered patient data
- **Grad-CAM Heatmaps**: Visual explanation of model predictions
- **Survival Rate Estimation**: AI-generated prognosis
- **Detailed Probabilities**: Confidence levels for each class

### 5. Medical Assistant Chat

1. From results page, click **"AI Doctor Assistant"**
2. Ask medical questions about the diagnosis
3. Get AI-powered responses based on analysis results

### 6. Dashboard

1. Navigate to **Dashboard** (http://localhost:8000/dashboard)
2. View **Analysis History**:
   - Total analyses performed
   - This month's analyses
3. **Download Reports**: Click download button on any analysis card
4. Reports include:
   - Patient information
   - Full diagnosis with confidence levels
   - Imaging study details
   - AI medical assistant summary
   - Clinical notes

---

## 📁 Project Structure

```
Brain Tumer Project/
├── webapp/                          # Main application
│   ├── app.py                      # FastAPI main application
│   ├── init_services.py            # Centralized Firebase & Groq initialization
│   ├── firebase_config.py          # Firebase initialization (deprecated)
│   ├── auth.py                     # Authentication helpers
│   ├── dashboard.html              # Dashboard UI
│   ├── login.html                  # Login/Signup page
│   ├── results_template.html       # Results display
│   ├── assistant_template.html     # AI assistant chat
│   └── firebase-service-account.json  # Firebase credentials
│
├── src/                             # Source code
│   ├── models/                     # Model architectures
│   │   ├── multimodal.py          # Multimodal fusion model
│   │   ├── resnet_multimodal.py   # ResNet-based model
│   │   └── unet.py                # U-Net for segmentation
│   │
│   ├── train/                      # Training scripts
│   │   ├── train_basic.py         # Basic training
│   │   ├── train_classification.py # Classification training
│   │   ├── train_multimodal.py    # Multimodal training
│   │   └── baseline_sklearn.py    # Sklearn baselines
│   │
│   ├── eval/                       # Evaluation utilities
│   │   ├── evaluate_basic.py      # Model evaluation
│   │   ├── explainability.py      # Grad-CAM implementation
│   │   └── metrics.py             # Performance metrics
│   │
│   ├── data/                       # Data processing
│   │   ├── loaders.py             # Dataset loaders
│   │   ├── multimodal_dataset.py  # Multimodal dataset
│   │   ├── preprocess.py          # Preprocessing utilities
│   │   └── augmentations.py       # Data augmentation
│   │
│   ├── inference/                  # Inference utilities
│   │   └── predict.py             # Prediction helpers
│   │
│   └── utils/                      # Utilities
│       ├── config.py              # Configuration
│       └── logging.py             # Logging setup
│
├── Dataset/                         # Medical imaging data
│   ├── Brain Tumor CT scan Images/
│   │   ├── Healthy/
│   │   └── Tumor/
│   └── Brain Tumor MRI images/
│       ├── Healthy/
│       └── Tumor/
│
├── models/                          # Trained model files
│   ├── model_binary.pth           # Binary classifier
│   ├── model_multiclass.pth       # Multi-class classifier
│   └── model_basic.pth            # Basic model
│
├── experiments/                     # Experiment logs
│   └── exp_001/
│       └── metrics.json           # Training metrics
│
├── notebooks/                       # Jupyter notebooks
│   ├── 01_exploration.ipynb       # Data exploration
│   ├── 02_preprocessing.ipynb     # Preprocessing
│   └── 03_experiments.ipynb       # Experiments
│
├── tools/                           # Utility scripts
│   ├── generate_manifest.py       # Dataset manifest
│   └── smoke_test_model.py        # Model testing
│
├── requirements.txt                 # Python dependencies
├── manifest.csv                     # Dataset manifest
├── train_models.py                 # Convenient model training runner
├── .env                            # Environment variables
├── .gitignore                      # Git ignore rules
└── README.md                       # This file
```

---

## 🎓 Model Training

### Prepare Dataset

Organize images in this structure:

```
Dataset/
├── Brain Tumor CT scan Images/
│   ├── Healthy/
│   │   ├── image001.jpg
│   │   └── ...
│   └── Tumor/
│       ├── image001.jpg
│       └── ...
└── Brain Tumor MRI images/
    ├── Healthy/
    └── Tumor/
```

### Train Binary Model (CT Scans)

```powershell
# Using the comprehensive ResNet50 training script
python src/train/train_resnet50.py \
  --data_dir "Dataset/Brain Tumor CT scan Images" \
  --model_type binary \
  --epochs 50 \
  --batch_size 32 \
  --lr 0.0001 \
  --output_dir .

# Or use the convenient runner script
python train_models.py --model binary
```

### Train Multiclass Model (MRI Scans)

```powershell
# Using the comprehensive ResNet50 training script
python src/train/train_resnet50.py \
  --data_dir "Dataset/Brain Tumor MRI images" \
  --model_type multiclass \
  --epochs 75 \
  --batch_size 16 \
  --lr 0.0001 \
  --output_dir .

# Or use the convenient runner script
python train_models.py --model multiclass
```

### Training Features

- **Architecture**: ResNet50 with ImageNet pretrained weights
- **Transfer Learning**: Freezes early layers, fine-tunes final 20 layers
- **Data Augmentation**: RandomCrop, RandomHorizontalFlip, RandomRotation, ColorJitter
- **80/20 Split**: Automatic train/validation split with random_split (seed=42)
- **Early Stopping**: Patience of 15 epochs to prevent overfitting
- **Learning Rate Scheduling**: ReduceLROnPlateau for adaptive learning
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1, AUC-ROC
- **Visualization**: Automatic generation of training curves and confusion matrix plots

### Training Parameters

| Parameter | Description | Default | Recommended |
|-----------|-------------|---------|-------------|
| `--epochs` | Training epochs | 10 | 20-30 |
| `--batch-size` | Batch size | 32 | 16-32 |
| `--lr` | Learning rate | 0.001 | 0.0001-0.001 |
| `--weight-decay` | L2 regularization | 0.0001 | 0.0001 |
| `--num-workers` | Data loading workers | 4 | 4-8 |

### Monitor Training

```powershell
# View training metrics
python -m json.tool experiments/exp_001/metrics.json
```

---

## 🔌 API Documentation

### Authentication Flow

**Public Endpoints:**
- `GET /` - Homepage
- `GET /login` - Login page
- `POST /api/login` - User authentication
- `POST /api/signup` - User registration

**Protected Endpoints** (require `Authorization: Bearer <token>` header):
- `POST /predict` - Analyze scans
- `GET /results` - View results
- `GET /chatbot` - Chat interface
- `GET /assistant` - AI assistant
- `POST /api/tumor-stage` - Tumor staging
- `POST /api/prognosis` - Survival prediction
- `POST /chat` - Chat message
- `GET /dashboard` - User dashboard
- `GET /api/dashboard` - Dashboard data

**Authentication Error Handling:**
- Returns `401 Unauthorized` with JSON: `{"detail": "Error message"}`
- Frontend intercepts 401 and redirects to login with modal message
- Modal displays authentication requirement with smooth UX
- Login form reveals after modal dismissal

### Authentication Endpoints

#### POST `/api/signup`
Create new user account

**Request:**
```json
{
  "email": "user@example.com",
  "password": "securepassword123"
}
```

**Response:**
```json
{
  "message": "User created successfully",
  "user_id": "abc123..."
}
```

#### POST `/api/login`
Login existing user

**Request:**
```json
{
  "email": "user@example.com",
  "password": "securepassword123"
}
```

**Response:**
```json
{
  "token": "eyJhbGciOiJSUzI1...",
  "user_id": "abc123..."
}
```

### Analysis Endpoints

#### POST `/predict`
Analyze brain scans

**Request** (multipart/form-data):
- `ct_file`: CT scan image (optional)
- `mri_file`: MRI scan image (optional)
- `metadata`: JSON string with patient info
- `Authorization`: Bearer token (header)

**Response:**
```json
{
  "final_prediction": "Benign",
  "final_probs": {
    "Healthy": "9.56%",
    "Benign": "86.40%",
    "Malignant": "4.04%"
  },
  "final_severity": {
    "label": "Moderate",
    "level": 2
  },
  "stage": "Two-stage (binary → multiclass)",
  "gradcam_ct": "data:image/png;base64,...",
  "gradcam_mri": "data:image/png;base64,...",
  "assistant_summary": "AI generated medical summary..."
}
```

#### GET `/api/dashboard`
Get user's analysis history

**Headers:**
- `Authorization`: Bearer token

**Response:**
```json
{
  "total_analyses": 15,
  "month_analyses": 5,
  "total_messages": 42,
  "analyses": [
    {
      "id": "analysis_id",
      "patient_name": "John Doe",
      "patient_age": "45",
      "final_prediction": "Benign",
      "created_at": "2025-11-23T10:30:00",
      "assistant_summary": "..."
    }
  ]
}
```

#### POST `/api/prognosis`
Get survival rate prediction

**Request:**
```json
{
  "patient_data": {
    "name": "John Doe",
    "age": 45,
    "gender": "Male"
  },
  "diagnosis": "Benign",
  "confidence": "86.40%"
}
```

**Response:**
```json
{
  "survival_rates": {
    "1_year": "95%",
    "5_year": "85%",
    "10_year": "75%"
  },
  "prognostic_factors": ["Age", "Tumor type", "Treatment"],
  "recommendations": ["Surgery", "Follow-up monitoring"]
}
```

---

## 🛠️ Troubleshooting

### Common Issues

#### 1. Import Errors (torch, torchvision)

**Problem:**
```
ModuleNotFoundError: No module named 'torch'
```

**Solution:**
```powershell
# Activate virtual environment
.\env\Scripts\Activate.ps1

# Reinstall PyTorch
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

#### 2. Firebase Authentication Failed

**Problem:**
```
Firebase initialized with errors: Could not find credentials
```

**Solution:**
- Verify `webapp/firebase-service-account.json` exists
- Check JSON file has valid Firebase credentials
- Ensure proper file permissions

#### 3. Model Not Found

**Problem:**
```
Model file not found: model_binary.pth
```

**Solution:**
```powershell
# Download pre-trained models or train new ones
python src/train/train_basic.py
```

#### 4. Groq API Rate Limit

**Problem:**
```
Groq API error: Rate limit exceeded
```

**Solution:**
- Wait a few minutes before retrying
- Upgrade to paid Groq plan for higher limits
- Implement request caching

#### 5. Port Already in Use

**Problem:**
```
Error: Address already in use
```

**Solution:**
```powershell
# Find process using port 8000
netstat -ano | findstr :8000

# Kill the process (replace PID)
taskkill /PID <process_id> /F

# Or use different port
python -m uvicorn webapp.app:app --port 8001
```

### Debug Mode

Enable detailed logging:

```powershell
# Set environment variable
$env:DEBUG="True"

# Run server
python -m uvicorn webapp.app:app --reload --log-level debug
```

### Check System Status

```powershell
# Verify Python version
python --version  # Should be 3.10+

# Check installed packages
pip list

# Test Firebase connection
python -c "from webapp.firebase_config import db; print('Firebase OK')"

# Test model loading
python -c "import torch; print('PyTorch version:', torch.__version__)"
```

---

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

### Development Setup

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Make changes and test thoroughly
4. Commit: `git commit -m 'Add amazing feature'`
5. Push: `git push origin feature/amazing-feature`
6. Open Pull Request

### Code Standards

- Follow PEP 8 style guide
- Add docstrings to functions
- Write unit tests for new features
- Update documentation

### Testing

```powershell
# Run tests
python -m pytest tests/

# Check code coverage
pytest --cov=webapp --cov=src
```

---

## 📄 License

This project is licensed for **research and educational purposes only**. 

**⚠️ Medical Disclaimer**: This software is NOT approved for clinical use. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for medical decisions.

---

## 🙏 Acknowledgments

- **Dataset**: Brain Tumor MRI/CT images from public medical datasets
- **Framework**: FastAPI, PyTorch, Firebase
- **AI Models**: ResNet architecture with transfer learning
- **LLM**: Groq API with LLama 3.3-70b-versatile
- **UI Framework**: Tailwind CSS, Font Awesome

---

## 📞 Support

For issues, questions, or contributions:

- **GitHub Issues**: [Report a bug](https://github.com/manojmahadevappa/Brain-Tumer-AI/issues)
- **Email**: manojmahadevappam@gmail.com
---

## 🗺️ Roadmap

### Version 2.0 (Upcoming)
- [ ] 3D volume analysis for MRI sequences
- [ ] Tumor segmentation with U-Net
- [ ] Multi-language support
- [ ] Mobile application (iOS/Android)
- [ ] DICOM file support
- [ ] Advanced visualization dashboard
- [ ] Integration with PACS systems
- [ ] Federated learning for privacy-preserving training


*Last Updated: November 23, 2025*
