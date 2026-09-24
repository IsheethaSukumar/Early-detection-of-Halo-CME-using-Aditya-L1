# Solar Halos Forecasting & Nowcasting - Full Execution Guide

This document contains instructions to run all components of the system:
1. **Frontend (Vite + React Dashboard)**
2. **Backend Microservice (FastAPI)**
3. **Real-time Telemetry Simulator / Stream Worker**

---

## 📁 Project Structure

```
Project-Code/
├── backend/
│   ├── Dataset/                 # Processed solar datasets
│   ├── ML_Model/                # FastAPI app, ML model files, results
│   └── 01_swis_cme_preprocessing.ipynb
└── frontend/                    # React 18 + Vite Dashboard components
```

---

## 1. Frontend (Vite React Dashboard)

### How to Run:

**Option A (From project root):**
```bash
node Project-Code/frontend/node_modules/vite/bin/vite.js Project-Code/frontend --port 5173
```

**Option B (From `Project-Code/frontend` folder):**
```bash
cd Project-Code/frontend
npm run dev
```

---

## 2. Backend Microservice (FastAPI API)

The backend runs on Python using FastAPI and Uvicorn.

### Step 1: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Start FastAPI Backend Server
Run Uvicorn from the **project root directory**:

```bash
python -m uvicorn Project-Code.backend.ML_Model.app:app --host 0.0.0.0 --port 8000 --reload
```

*Interactive API Docs:* `http://localhost:8000/docs`

---

## 3. Real-Time Telemetry Data Simulator (Optional)

To simulate streaming space weather telemetry from Aditya-L1 and feed live data to the database/backend:

Run from the **project root directory**:
```bash
python Project-Code/backend/ML_Model/simulate_telemetry.py
```

---

## 4. Running via Docker (Full Stack)

If you have Docker & Docker Compose installed, launch everything in one command from the project root:

```bash
docker-compose up --build
```
