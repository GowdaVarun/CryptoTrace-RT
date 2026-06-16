import os
import sys
import json
import uuid
import shutil
from contextlib import asynccontextmanager

import pandas as pd
import xgboost as xgb
import shap
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Add scripts directory to path to import existing extraction logic
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from feature_extraction import extract_features

try:
    from sandbox import empty_dynamic_features, has_static_detection, run_in_docker_sandbox
except ImportError:
    from backend.sandbox import empty_dynamic_features, has_static_detection, run_in_docker_sandbox

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'crypto_detector_xgboost.json')
TOP10_PATH = os.path.join(BASE_DIR, '..', 'top10_features.json')
TEMP_DIR   = os.path.join(BASE_DIR, 'temp_uploads')

os.makedirs(TEMP_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# App state (populated at startup)
# ─────────────────────────────────────────────────────────────────────────────

model:            xgb.XGBClassifier | None = None
explainer:        shap.TreeExplainer | None = None
expected_columns: list[str] = []


# Use the modern lifespan pattern instead of the deprecated @app.on_event
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, explainer, expected_columns

    print("Loading XGBoost model...")
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)

    print("Initializing SHAP explainer...")
    explainer = shap.TreeExplainer(model)

    print("Loading top-10 feature list...")
    with open(TOP10_PATH, 'r') as f:
        top10_data = json.load(f)
        expected_columns = top10_data['features']

    print(f"Ready — {len(expected_columns)} features loaded.")
    yield
    # (shutdown logic would go here if needed)


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="CryptoTrace Runtime Analysis API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/analyze")
async def analyze_binary(file: UploadFile = File(...)):
    if model is None or explainer is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    file_id     = str(uuid.uuid4())
    safe_name   = os.path.basename(file.filename or "upload.bin")
    temp_path   = os.path.join(TEMP_DIR, f"{file_id}_{safe_name}")

    try:
        # ── Save upload ───────────────────────────────────────────────────────
        with open(temp_path, "wb") as buf:
            shutil.copyfileobj(file.file, buf)
        os.chmod(temp_path, 0o755)

        # ── 1. Static feature extraction ──────────────────────────────────────
        print(f"[{safe_name}] Extracting static features...")
        try:
            static_features = extract_features(temp_path)
            if not static_features:
                raise ValueError("LIEF returned empty feature set.")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Static analysis failed: {exc}")

        # ── 2. Dynamic feature extraction ─────────────────────────────────────
        # Never execute uploads directly on the host.
        # If static crypto indicators are already present we skip runtime
        # execution entirely — there is no value in running a known-crypto binary.
        if has_static_detection(static_features):
            print(f"[{safe_name}] Static indicators found — skipping runtime execution.")
            dyn_features = empty_dynamic_features()
            sandbox = {
                "mode":         "skipped",
                "status":       "static_detection_present",
                "image":        None,
                "perf_status":  "not_run",
                "strace_status":"not_run",
                "return_code":  None,
            }
        else:
            print(f"[{safe_name}] No static indicators — launching Docker sandbox.")
            dyn_features, sandbox = run_in_docker_sandbox(temp_path)

        # ── 3. Build inference DataFrame ──────────────────────────────────────
        all_features = {**static_features, **dyn_features}
        row_data     = {col: all_features.get(col, 0.0) for col in expected_columns}
        df_infer     = pd.DataFrame([row_data])

        # ── 4. Predict ────────────────────────────────────────────────────────
        prob       = float(model.predict_proba(df_infer)[0][1])
        prediction = "Crypto" if prob > 0.5 else "Non-Crypto"

        # ── 5. SHAP explanation ───────────────────────────────────────────────
        shap_values = explainer.shap_values(df_infer)

        # Normalise to a 1-D array regardless of shap version / output format
        if isinstance(shap_values, list):
            shap_values = shap_values[1][0]   # multi-class list format
        elif shap_values.ndim > 1:
            shap_values = shap_values[0]       # 2-D array, single sample

        shap_data = [
            {
                "feature":    col,
                "shap_value": float(shap_values[i]),
                "raw_value":  float(row_data[col]),
            }
            for i, col in enumerate(expected_columns)
        ]
        shap_data.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        return {
            "filename":          file.filename,
            "prediction":        prediction,
            "confidence":        prob,
            "sandbox":           sandbox,
            "top_features_shap": shap_data[:15],
            "all_features":      row_data,
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/debug")
async def debug_binary(file: UploadFile = File(...)):
    """
    Same pipeline as /analyze but returns a detailed step-by-step breakdown
    so you can see exactly what happened at each stage.
    Use this to verify Docker is running and dynamic features are being extracted.
    """
    file_id   = str(uuid.uuid4())
    safe_name = os.path.basename(file.filename or "upload.bin")
    temp_path = os.path.join(TEMP_DIR, f"{file_id}_{safe_name}")

    report = {
        "filename": file.filename,
        "steps": {}
    }

    try:
        with open(temp_path, "wb") as buf:
            shutil.copyfileobj(file.file, buf)
        os.chmod(temp_path, 0o755)

        # ── Step 1: Static features ───────────────────────────────────────────
        try:
            static_features = extract_features(temp_path)
            report["steps"]["1_static_extraction"] = {
                "status": "ok",
                "feature_count": len(static_features),
                "features": static_features,
            }
        except Exception as exc:
            report["steps"]["1_static_extraction"] = {"status": "error", "detail": str(exc)}
            return report

        # ── Step 2: Static detection gate ─────────────────────────────────────
        triggered = {
            k: float(static_features.get(k, 0.0))
            for k in (
                "n_crypto_imports", "n_crypto_import_categories", "crypto_import_ratio",
                "has_crypto_library", "n_crypto_libraries", "crypto_constant_hits",
                "rodata_crypto_hits", "n_crypto_strings", "crypto_string_ratio",
            )
        }
        static_hit = any(v > 0 for v in triggered.values())
        report["steps"]["2_static_detection_gate"] = {
            "static_crypto_detected": static_hit,
            "indicator_values": triggered,
            "decision": "SKIP sandbox (static hit)" if static_hit else "PROCEED to sandbox",
        }

        # ── Step 3: Docker sandbox ────────────────────────────────────────────
        if static_hit:
            dyn_features = empty_dynamic_features()
            sandbox_meta = {"mode": "skipped", "reason": "static_detection_present"}
            report["steps"]["3_docker_sandbox"] = {"status": "skipped", "reason": "static_detection_present"}
        else:
            dyn_features, sandbox_meta = run_in_docker_sandbox(temp_path)
            report["steps"]["3_docker_sandbox"] = {
                "status": "ran",
                "metadata": sandbox_meta,
                "dynamic_features_extracted": dyn_features,
                "any_nonzero": any(v > 0 for v in dyn_features.values()),
            }

        # ── Step 4: Model input ───────────────────────────────────────────────
        all_features = {**static_features, **dyn_features}
        row_data     = {col: all_features.get(col, 0.0) for col in expected_columns}
        report["steps"]["4_model_input"] = {
            "expected_columns": expected_columns,
            "values_fed_to_model": row_data,
            "note": "If all dynamic features are 0 here, Docker either didn't run or failed silently.",
        }

        # ── Step 5: Prediction ────────────────────────────────────────────────
        if model:
            df_infer   = pd.DataFrame([row_data])
            prob       = float(model.predict_proba(df_infer)[0][1])
            prediction = "Crypto" if prob > 0.5 else "Non-Crypto"
            report["steps"]["5_prediction"] = {
                "prediction": prediction,
                "confidence": prob,
            }
        else:
            report["steps"]["5_prediction"] = {"status": "model_not_loaded"}

        return report

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)