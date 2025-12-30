# MediscopeAI - System Resource Utilization Report

**Project Version:** 1.0  
**Report Date:** December 30, 2025  
**Analysis Purpose:** Determine deployment server requirements

---

## 📊 Executive Summary

### Current Deployment Target
- **RAM:** 1GB
- **CPU:** 1 vCPU
- **Storage:** 35GB NVMe SSD
- **Bandwidth:** 1000GB/month transfer

### Verdict
⚠️ **NOT RECOMMENDED** - The target specs are insufficient for production deployment.

**Minimum Recommended Specs:**
- **RAM:** 2-4GB (minimum 2GB)
- **CPU:** 2 vCPUs
- **Storage:** 50GB SSD
- **Bandwidth:** 1000GB/month (acceptable)

---

## 🔍 Detailed Resource Analysis

### 1. Storage Requirements

#### Application Files
| Component | Size | Description |
|-----------|------|-------------|
| PyTorch Models | **172.85 MB** | Binary (42.71MB) + Multiclass (87.43MB) + Basic (42.71MB) |
| Web Application | 0.55 MB | FastAPI app, HTML templates, static files |
| Source Code | 0.10 MB | ML training scripts, utilities |
| **Subtotal** | **173.50 MB** | Core application |

#### Python Dependencies (Estimated)
| Package Category | Estimated Size |
|------------------|----------------|
| PyTorch (CPU version) | ~200 MB |
| NumPy + SciPy | ~50 MB |
| FastAPI + Uvicorn | ~20 MB |
| Firebase Admin SDK | ~30 MB |
| Image Processing (PIL, OpenCV, scikit-image) | ~100 MB |
| Other dependencies (27 total) | ~100 MB |
| **Total Dependencies** | **~500 MB** |

#### Runtime & System Files
- Python 3.10+ runtime: ~150 MB
- System libraries and cache: ~200 MB
- Log files and temporary files: ~100 MB

#### User Data (Firebase Firestore - Cloud Storage)
- Analysis results stored in Firebase (not on local disk)
- Temporary image uploads: ~10-50 MB
- Session data and cache: ~50 MB

#### **Total Storage Required**
```
Core Application:     173 MB
Python Environment:   500 MB
System Overhead:      450 MB
User Data Cache:      100 MB
Growth Buffer:        300 MB
─────────────────────────────
TOTAL:               ~1.5 GB
Recommended:          3-5 GB (with deployment artifacts)
```

**Storage Verdict:** ✅ 35GB SSD is **MORE THAN SUFFICIENT**

---

### 2. RAM Requirements

#### Base Memory Consumption
| Process | Memory Usage |
|---------|--------------|
| Python Interpreter | 50-80 MB |
| FastAPI + Uvicorn | 80-120 MB |
| Firebase Admin SDK | 50-80 MB |
| **Base Runtime** | **180-280 MB** |

#### Model Loading (One-time on startup)
| Model | Memory Footprint |
|-------|------------------|
| Binary Model (ResNet50) | ~170 MB |
| Multiclass Model (ResNet50) | ~350 MB |
| Model Caching | ~100 MB |
| **Total Models in Memory** | **~620 MB** |

#### Per-Request Memory (Inference)
| Operation | Memory Spike |
|-----------|--------------|
| Image Upload & Preprocessing | 20-40 MB |
| PyTorch Inference | 150-200 MB |
| Grad-CAM Generation | 50-100 MB |
| JSON Response Assembly | 10-20 MB |
| **Per Request Peak** | **230-360 MB** |

#### Concurrent User Handling
**With 1GB RAM:**
- Base + Models: ~900 MB
- Available for requests: ~100 MB
- **Concurrent Users:** 0-1 user max
- **Result:** System will OOM (Out of Memory) with multiple users

**With 2GB RAM:**
- Base + Models: ~900 MB
- Available for requests: ~1.1 GB
- **Concurrent Users:** 3-4 users
- **Result:** Acceptable for light traffic

**With 4GB RAM:**
- Base + Models: ~900 MB
- Available for requests: ~3.1 GB
- **Concurrent Users:** 8-10 users
- **Result:** Good for moderate traffic

**RAM Verdict:** ❌ 1GB is **INSUFFICIENT**  
**Minimum:** 2GB RAM  
**Recommended:** 4GB RAM

---

### 3. CPU Requirements

#### Compute-Intensive Operations
| Operation | CPU Demand | Duration |
|-----------|------------|----------|
| Model Loading (startup) | High | 5-10 seconds |
| Image Preprocessing | Low-Medium | 0.1-0.2 seconds |
| ResNet50 Inference (CPU) | **Very High** | 1-3 seconds |
| Grad-CAM Generation | High | 0.5-1 second |
| Groq API Call (external) | Low | 0.5-2 seconds |
| JSON Serialization | Low | 0.05 seconds |
| **Total per request** | - | **2-6 seconds** |

#### Single CPU Limitations
- **1 vCPU:** Can only handle 1 inference at a time
- **Concurrent requests:** Will queue, causing delays
- **Response time:** 2-6 seconds per user (serial processing)
- **Throughput:** ~10-30 requests/minute maximum

#### Multi-CPU Benefits
- **2 vCPUs:** Can process 2 concurrent inferences
- **Throughput:** ~40-60 requests/minute
- **User Experience:** Significantly improved

**CPU Verdict:** ⚠️ 1 vCPU is **MARGINAL**  
**Minimum:** 1 vCPU (low traffic only)  
**Recommended:** 2 vCPUs

---

### 4. Network Bandwidth Requirements

#### Average Request Breakdown
| Data Transfer Type | Size | Notes |
|-------------------|------|-------|
| **Inbound (Upload)** |
| CT/MRI Image Upload | 0.5-3 MB | Average 1-2 MB per image |
| Patient Metadata | 1-5 KB | JSON payload |
| Authentication Tokens | 2-5 KB | JWT tokens |
| **Per Request Inbound** | **1-6 MB** | - |
| **Outbound (Download)** |
| Analysis Results JSON | 50-200 KB | Predictions, probabilities |
| Grad-CAM Heatmaps (2x) | 200-500 KB | Base64 encoded images |
| Original Scans (2x) | 1-4 MB | For display |
| HTML/CSS/JS | 100-300 KB | Page assets |
| **Per Request Outbound** | **1.5-5 MB** | - |

#### Monthly Bandwidth Estimation
**Scenario 1: Light Traffic (100 users/month)**
- 100 analyses × 6 MB average = 600 MB inbound
- 100 analyses × 5 MB average = 500 MB outbound
- Dashboard access (500 page views) = 150 MB
- **Total:** ~1.25 GB/month

**Scenario 2: Moderate Traffic (500 users/month)**
- 500 analyses × 6 MB = 3 GB inbound
- 500 analyses × 5 MB = 2.5 GB outbound
- Dashboard access (2000 page views) = 600 MB
- **Total:** ~6.1 GB/month

**Scenario 3: Heavy Traffic (2000 users/month)**
- 2000 analyses × 6 MB = 12 GB inbound
- 2000 analyses × 5 MB = 10 GB outbound
- Dashboard + static assets = 3 GB
- **Total:** ~25 GB/month

**Bandwidth Verdict:** ✅ 1000GB/month is **EXCELLENT**  
Can support **30,000-40,000 analyses/month**

---

## 🎯 Deployment Recommendations

### Tier 1: Minimum Viable (Small Scale)
**Suitable for:** Demo, personal use, <100 users/month
```
RAM:       2 GB
CPU:       1 vCPU
Storage:   10 GB SSD
Bandwidth: 100 GB/month
Cost:      ~$6-12/month (DigitalOcean, Linode)
```
**Limitations:**
- Single concurrent user
- Slow response times (3-6 seconds)
- May crash under load

---

### Tier 2: Recommended Production (Medium Scale)
**Suitable for:** Clinical deployment, <1000 users/month
```
RAM:       4 GB
CPU:       2 vCPUs
Storage:   25 GB SSD
Bandwidth: 500 GB/month
Cost:      ~$24-48/month (DigitalOcean, Linode)
```
**Benefits:**
- 3-5 concurrent users
- Fast response times (2-4 seconds)
- Stable under moderate load
- Room for growth

---

### Tier 3: Production Plus (Large Scale)
**Suitable for:** Hospital networks, >2000 users/month
```
RAM:       8 GB
CPU:       4 vCPUs
Storage:   50 GB SSD
Bandwidth: 1000 GB/month
Cost:      ~$48-96/month (DigitalOcean, Linode)
```
**Benefits:**
- 8-12 concurrent users
- Optimal response times (1-2 seconds)
- High availability
- Professional-grade performance

---

## 📈 Performance Optimization Tips

### 1. Memory Optimization
- **Model Quantization:** Reduce model size by 50-75% (PyTorch quantization)
- **Lazy Loading:** Load models on first request (saves 600MB on startup)
- **Swap Configuration:** Add 2GB swap space for memory spikes

### 2. CPU Optimization
- **Batch Processing:** Queue requests and process in small batches
- **ONNX Runtime:** Convert models to ONNX for 2-3x faster inference
- **Multi-threading:** Use worker processes for concurrent requests

### 3. Storage Optimization
- **Model Compression:** Use `torch.save()` with compression
- **Log Rotation:** Limit log file sizes to 50MB
- **Image Caching:** Cache preprocessed images temporarily

### 4. Bandwidth Optimization
- **Image Compression:** Compress Grad-CAM images to WebP format
- **CDN:** Use Cloudflare (free) for static assets
- **Lazy Loading:** Load images on demand in UI

---

## 🚨 Critical Warnings for 1GB/1vCPU Deployment

### Immediate Issues You'll Face:
1. **Out of Memory Crashes**
   - Models alone consume ~620MB
   - Base runtime: ~280MB
   - Total: 900MB (90% of available RAM)
   - First inference attempt: **CRASH**

2. **Swap Thrashing**
   - If swap is configured, system will be extremely slow
   - Response times: 30-60 seconds per request
   - CPU will be 100% busy swapping memory

3. **No Concurrent Users**
   - Second user attempt will crash the server
   - No graceful degradation

4. **Production Instability**
   - Random crashes under load
   - Requires constant restarts
   - Poor user experience

---

## ✅ Final Recommendation

### For Your Target Specs (1GB/1vCPU/35GB)
**Status:** ❌ **NOT PRODUCTION READY**

### Upgrade Path Options:

#### Option A: Minimum Upgrade (Budget)
```
FROM: 1GB/1vCPU   → TO: 2GB/2vCPU
Cost: +$10-15/month
Result: Functional but slow (3-6 second response)
```

#### Option B: Recommended Upgrade
```
FROM: 1GB/1vCPU   → TO: 4GB/2vCPU
Cost: +$20-35/month
Result: Professional performance (2-4 second response)
```

#### Option C: Development Only
Keep current specs for:
- Local development
- Testing
- Demo purposes only
- **NOT for production deployment**

---

## 📊 Resource Monitoring Setup

### Essential Monitoring
```bash
# Memory monitoring
free -h
htop

# Disk usage
df -h
du -sh /path/to/app

# Network stats
iftop
vnstat

# Application logs
tail -f /var/log/mediscope/app.log
```

### Performance Metrics to Track
- **Response Time:** Should be <5 seconds
- **Memory Usage:** Should stay <80% capacity
- **CPU Usage:** Average <70%, peaks <90%
- **Error Rate:** Should be <1%
- **Uptime:** Should be >99.5%

---

## 🎓 Summary Table

| Resource | Current Target | Minimum Viable | Recommended | Status |
|----------|----------------|----------------|-------------|--------|
| **RAM** | 1 GB | 2 GB | 4 GB | ❌ Insufficient |
| **CPU** | 1 vCPU | 1 vCPU | 2 vCPUs | ⚠️ Marginal |
| **Storage** | 35 GB | 10 GB | 25 GB | ✅ Excellent |
| **Bandwidth** | 1000 GB | 100 GB | 500 GB | ✅ Excellent |

**Overall Verdict:** Upgrade RAM to 2GB minimum (4GB recommended) for stable production deployment.

---

**Report Generated:** December 30, 2025  
**Next Review:** After 1 month of production metrics
