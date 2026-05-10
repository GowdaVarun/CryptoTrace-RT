import os
import sys
import json
import uuid
import shutil
import subprocess
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add scripts directory to path to import existing extraction logic
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from feature_extraction import extract_features
from dynamic_analysis import run_perf, run_strace

app = FastAPI(title="CryptoTrace Runtime Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'results', 'crypto_detector_xgboost.json')
STATIC_CSV = os.path.join(BASE_DIR, '..', 'dataset', 'binary_features.csv')
DYN_CSV = os.path.join(BASE_DIR, '..', 'dataset', 'dynamic_features.csv')
TEMP_DIR = os.path.join(BASE_DIR, 'temp_uploads')

os.makedirs(TEMP_DIR, exist_ok=True)

# Global model and explainer
model = None
explainer = None
expected_columns = []

@app.on_event("startup")
def load_resources():
    global model, explainer, expected_columns
    print("Loading XGBoost model...")
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    
    # Initialize SHAP explainer
    print("Initializing SHAP explainer...")
    explainer = shap.TreeExplainer(model)
    
    # The XGBoost model was trained on the top 10 features ONLY.
    top10_path = os.path.join(BASE_DIR, '..', 'results', 'top10_features.json')
    with open(top10_path, 'r') as f:
        top10_data = json.load(f)
        expected_columns = top10_data['features']
        
    print(f"Loaded {len(expected_columns)} expected feature columns from Top 10.")

@app.post("/analyze")
async def analyze_binary(file: UploadFile = File(...)):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
        
    file_id = str(uuid.uuid4())
    temp_path = os.path.join(TEMP_DIR, f"{file_id}_{file.filename}")
    
    try:
        # Save file
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Make executable
        os.chmod(temp_path, 0o755)
        
        # 1. Extract Static Features
        print(f"Extracting static features for {file.filename}...")
        try:
            static_features = extract_features(temp_path)
            if not static_features:
                raise Exception("LIEF failed to parse binary.")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Static analysis failed: {str(e)}")
            
        # 2. Extract Dynamic Features
        print(f"Extracting dynamic features for {file.filename}...")
        try:
            # Brief execution for exec time
            import time
            start = time.time()
            subprocess.run([temp_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2)
            exec_time = time.time() - start
        except:
            exec_time = 0.05
            
        perf_metrics = run_perf(temp_path) or {
            "dyn_instructions": 0.0, "dyn_cycles": 0.0, 
            "dyn_branches": 0.0, "dyn_branch_misses": 0.0
        }
        strace_metrics = run_strace(temp_path) or {
            "dyn_total_syscalls": 0.0, "dyn_unique_syscalls": 0.0,
            "dyn_getrandom_calls": 0.0, "dyn_read_calls": 0.0, "dyn_write_calls": 0.0
        }
        
        dyn_features = {"dyn_exec_time": exec_time}
        dyn_features.update(perf_metrics)
        dyn_features.update(strace_metrics)
        dyn_features['dyn_ipc'] = dyn_features['dyn_instructions'] / max(dyn_features['dyn_cycles'], 1)
        dyn_features['dyn_branch_miss_ratio'] = dyn_features['dyn_branch_misses'] / max(dyn_features['dyn_branches'], 1)
        
        # Merge all features
        all_features = {**static_features, **dyn_features}
        
        # Construct dataframe in exact column order
        row_data = {}
        for col in expected_columns:
            row_data[col] = all_features.get(col, 0.0)
            
        df_infer = pd.DataFrame([row_data])
        
        # 3. Predict
        prob = float(model.predict_proba(df_infer)[0][1])
        prediction = "Crypto" if prob > 0.5 else "Non-Crypto"
        
        # 4. SHAP Explanation
        shap_values = explainer.shap_values(df_infer)
        
        # Ensure shap_values is a 1D array for a single sample
        if isinstance(shap_values, list):
            shap_values = shap_values[1][0]  # If multi-class format
        elif len(shap_values.shape) > 1:
            shap_values = shap_values[0]
            
        # Pair feature names with their SHAP values and raw values
        shap_data = []
        for i, col in enumerate(expected_columns):
            shap_data.append({
                "feature": col,
                "shap_value": float(shap_values[i]),
                "raw_value": float(row_data[col])
            })
            
        # Sort by absolute SHAP value
        shap_data.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        top_shap = shap_data[:15] # Return top 15 for UI
        
        return {
            "filename": file.filename,
            "prediction": prediction,
            "confidence": prob,
            "top_features_shap": top_shap,
            "all_features": row_data
        }
        
    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
