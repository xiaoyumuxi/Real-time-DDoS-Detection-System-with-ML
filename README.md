# DDoS Detection System - CyberDefense IDS Shield

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Performance Metrics](#performance-metrics)
- [Comprehensive Testing Checklist](#comprehensive-testing-checklist)
- [Metrics Justification](#metrics-justification)
- [Environment Setup](#environment-setup)
- [Usage Guide](#usage-guide)
- [API Documentation](#api-documentation)
- [Project Structure](#project-structure)

---
## 🛠️ Environment Setup

### Prerequisites

- **Python**: 3.8 or higher
- **Node.js**: 16.x or higher (for frontend)
- **npm**: 8.x or higher
- **Git**: For version control
- **Operating System**: Windows, macOS, or Linux

### Step 1: Backend Setup (Python/Flask)

#### 1.1 Create Virtual Environment

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

#### 1.2 Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- Flask (web framework)
- Flask-CORS (cross-origin support)
- scikit-learn (machine learning)
- pandas (data manipulation)
- numpy (numerical computing)
- joblib (model serialization)

### Step 2: Prepare Training Data

#### 2.1 Download Dataset

Download the CICIDS2017 dataset (Wednesday workingHours traffic):
- **Source**: [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html)
- **Required File**: `Wednesday-workingHours.pcap_ISCX.csv`

#### 3.2 Place Dataset

```bash
mkdir -p data
# Place Wednesday-workingHours.pcap_ISCX.csv in the data/ directory
```

**Expected file structure:**
```
SCC252/
├── data/
│   └── Wednesday-workingHours.pcap_ISCX.csv
```

### Step 3: Train the Model

```bash
python trainning.py
```

**Expected output:**
```
正在读取数据...
数据读取成功，原始形状: (692703, 79)
正在清理数据...
清理后形状: (692703, 79)
正在进行多分类标签编码...
...
测试集整体准确率: 0.9993
测试集整体精确度: 0.9993
测试集[BENIGN]召回率: 0.9993
测试集假阳性率: 0.0005
测试集AUC: 0.9999
🎉 任务完成！
```

**Generated files in `models/` directory:**
- `ddos_rf_model.joblib` - Trained Random Forest model
- `ddos_scaler.joblib` - StandardScaler for feature normalization
- `ddos_label_encoder.joblib` - Label encoder for attack types
- `ddos_feature_columns.joblib` - List of feature column names
- `ddos_performance.json` - Performance metrics

### Step 4: Start Backend Server

```bash
python app.py
```

**Expected output:**
```
INFO:werkzeug:WARNING: This is a development server.
 * Running on http://127.0.0.1:5050
INFO:app:✅ Model components loaded successfully.
INFO:app:✅ Performance metrics loaded.
```

**Verify backend is running:**
```bash
curl http://127.0.0.1:5050/health
# Expected: {"status":"healthy","model_loaded":true}
```

### Step 5: Frontend Setup (Vue.js)

#### 5.1 Navigate to Frontend Directory

```bash
cd template
```

#### 5.2 Install Node Dependencies

```bash
npm install
```

#### 5.3 Start Development Server

```bash
npm run dev
```

**Expected output:**
```
VITE v6.2.0  ready in 500 ms

➜  Local:   http://localhost:5173/
➜  Network: use --host to expose
```

### Step 6: Access the Application

1. **Open browser** and navigate to: `http://localhost:5173`
2. **Backend API** is available at: `http://127.0.0.1:5050`

**You should see:**
- Dashboard with performance metrics
- Real-time detection interface
- Alert history panel
- Model statistics radar chart

### Step 7: Verify Setup

#### 7.1 Test Prediction Endpoint

```bash
curl -X POST http://127.0.0.1:5050/api/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [54865,3,2,0,12,0,6,6,6.0,0.0,0,0,0.0,0.0,4000000.0,666666.6667,3.0,0.0,3,3,3,3.0,0.0,3,3,0,0.0,0.0,0,0,0,0,0,0,40,0,666666.6667,0.0,6,6,6.0,0.0,0.0,0,0,0,0,1,0,0,0,0,9.0,6.0,0.0,40,0,0,0,0,0,0,2,12,0,0,33,-1,1,20,0.0,0.0,0,0,0.0,0.0,0,0]}'
```

#### 8.2 Run Example Script

```bash
cd examples
python run_sample.py
```

**Expected output:**
```
--- 模拟网站/API 接口返回结果 ---
{
    "status": "success",
    "predicted_label": "BENIGN",
    "confidence": 0.98,
    "threat_level": "None",
    ...
}
```


## 📖 Usage Guide

### Running Sample Detection

```bash
cd examples
python run_sample.py
```

### Making Predictions via API

```python
import requests

# Prepare feature data (78 features required)
features = [54865, 3, 2, 0, 12, ...] # 78 values

# Send prediction request
response = requests.post(
    'http://127.0.0.1:5050/api/predict',
    json={'features': features}
)

result = response.json()
print(f"Prediction: {result['predicted_label']}")
print(f"Confidence: {result['confidence']}")
print(f"Threat Level: {result['threat_level']}")
```

### Retraining the Model

```bash
# Upload new CSV data via web interface
# Or use API:
curl -X POST http://127.0.0.1:5050/api/upload-and-retrain \
  -F "files=@new_data.csv"
```

### Viewing Detection History

Access the frontend dashboard at `http://localhost:5173` to view:
- Real-time attack detection
- Historical alerts
- Model performance metrics
- Attack frequency analysis

---

## 🔌 API Documentation

### Endpoints Overview

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/api/predict` | Classify network traffic |
| GET | `/api/alerts` | Get recent alerts |
| GET | `/api/history` | Get detection history |
| GET | `/api/performance` | Get model metrics |
| GET | `/api/stream` | Get attack stream samples |
| GET | `/api/random` | Generate random test data |
| POST | `/api/upload-and-retrain` | Retrain model with new data |

### Detailed API Reference

#### `POST /api/predict`

Classify network traffic features.

**Request:**
```json
{
  "features": [78 numerical values]
}
```

**Response:**
```json
{
  "status": "success",
  "predicted_label": "DoS Hulk",
  "confidence": 0.95,
  "threat_level": "High",
  "probabilities": {
    "DoS Hulk": 0.95,
    "BENIGN": 0.03,
    "DoS Slowloris": 0.02
  },
  "timestamp": "2025-12-14 10:30:45"
}
```

#### `GET /api/performance`

Get current model performance metrics.

**Response:**
```json
{
  "accuracy": 0.9993,
  "precision": 0.9993,
  "recall": 0.9993,
  "FPR": 0.0005,
  "auc": 0.9999
}
```

---

## 📁 Project Structure

```
SCC252/
├── app.py                          # Flask backend server
├── trainning.py                    # Model training script
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── data/                           # Training datasets
│   └── Wednesday-workingHours.pcap_ISCX.csv
├── models/                         # Trained model files
│   ├── ddos_rf_model.joblib
│   ├── ddos_scaler.joblib
│   ├── ddos_label_encoder.joblib
│   ├── ddos_feature_columns.joblib
│   └── ddos_performance.json
├── examples/                       # Usage examples
│   ├── run_sample.py
│   └── retrain_with_new_data.py
├── template/                       # Vue.js frontend
│   ├── components/
│   │   ├── RadarChart.vue
│   │   └── StatCard.vue
│   ├── services/
│   │   └── api.ts
│   ├── App.vue
│   ├── main.ts
│   ├── package.json
│   └── vite.config.ts
└── ddos_detection.db              # SQLite database
```

---





