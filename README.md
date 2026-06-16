# 🔐 CryptoTrace-RT: Real-Time Crypto Binary Detection

**CryptoTrace-RT** is a high-fidelity machine learning pipeline and interactive web dashboard for detecting cryptographic algorithms within compiled Linux ELF binaries. 

It utilizes a **Hybrid Analysis Approach**, fusing traditional static structural features (extracted via LIEF) with zero-overhead micro-architectural dynamic profiling (extracted via Linux `perf` and `strace`). By tracking hardware performance counters like **Branch Miss Ratios** and **Instructions Per Cycle (IPC)**, the model successfully bypasses static obfuscation techniques.

## 🚀 The Web Dashboard

To make the research tangible, this repository includes a full-stack web application. You can drag-and-drop any ELF binary and receive a real-time classification along with a **SHAP Interpretability** bar chart explaining exactly why the model made its decision.

### Running the Dashboard

You will need two separate terminal windows to run the API and the UI.

**1. Start the FastAPI Backend:**
```bash
cd /home/varun/varun/CryptoTrace-RT/
source venv/bin/activate
pip install -r backend/requirements.txt
docker build -f backend/Dockerfile.sandbox -t cryptotrace-sandbox:latest backend
cd backend
python3 app.py
```

**2. Start the React Frontend:**
```bash
cd /home/varun/varun/CryptoTrace-RT/frontend
npm install
npm run dev
```
Navigate to `http://localhost:5173` to interact with the dashboard.

*(Note: The backend dynamically profiles binaries using hardware performance counters. If you receive zeros for instruction metrics, ensure you have temporarily lowered the `perf` restriction by running: `sudo sysctl -w kernel.perf_event_paranoid=-1`)*

### Runtime Sandbox

Uploaded binaries are not executed directly on the host. The backend first performs static feature extraction. If explicit static crypto indicators are present, runtime execution is skipped. If no static indicators are present, the binary is run in a locked-down Docker container with network disabled, dropped Linux capabilities, a read-only root filesystem, and CPU/memory/PID limits.

The default sandbox image is `cryptotrace-sandbox:latest`. The included `backend/Dockerfile.sandbox` adds `strace` and `perf` so dynamic metrics can be collected when the host permits them:

```bash
docker build -f backend/Dockerfile.sandbox -t cryptotrace-sandbox:latest backend
```

If Docker or the configured image is unavailable, the API returns zero dynamic features with sandbox metadata instead of falling back to host execution.

---

## 📊 Model Performance

The Extreme Gradient Boosting (XGBoost) classifier was trained on a dataset of 316 binaries, evaluating 92 distinct static and dynamic features.

| Metric | Test Set Score |
|--------|----------|
| **Accuracy** | 96.88% |
| **F1 Score** | 96.77% |
| **ROC AUC** | 99.51% |
| **MCC** | 93.93% |

**Overfitting gap: 0.74%** (Train acc 97.62% vs Test acc 96.88%) ✅

---

## 🥇 Top 10 Discriminative Features

The introduction of dynamic micro-architectural profiling heavily displaced traditional static heuristics. Crypto loops are highly mathematically predictable, resulting in exceptionally low branch miss ratios compared to standard software.

| Rank | Feature | Type | What it Captures |
|------|---------|------|------------------|
| 1 | `crypto_string_ratio` | Static | Ratio of cryptographic strings (aes, sha, encrypt) |
| 2 | `sec_rodata_entropy` | Static | High entropy in read-only data often indicates S-boxes |
| 3 | `dyn_branch_misses` | **Dynamic** | Crypto loops mathematically prevent branch predictor misses |
| 4 | `dyn_branch_miss_ratio`| **Dynamic** | Ratio of missed branches; exceedingly low for crypto |
| 5 | `dyn_cycles` | **Dynamic** | Total CPU cycles executed during the harness timeout |
| 6 | `dyn_branches` | **Dynamic** | Total conditional branches evaluated by the CPU |
| 7 | `dyn_instructions` | **Dynamic** | Raw instruction volume executing inside the pipeline |
| 8 | `dyn_ipc` | **Dynamic** | Instructions Per Cycle; crypto maths run highly efficiently |
| 9 | `text_xor_density` | Static | XOR instruction density in `.text`; core to block ciphers |
| 10 | `avg_string_len` | Static | Structural indicator of packed/obfuscated code |

---

## 📂 Project Structure

- `backend/` — The FastAPI inference server (handles LIEF parsing, `perf`/`strace` execution, and SHAP calculation).
- `frontend/` — The Vite/React application with a custom Glassmorphism Cyberpunk aesthetic.
- `scripts/` — The original data generation and model training pipeline.
  - `build_dataset.py` (Generates 316 C binaries).
  - `feature_extraction.py` (Static analysis).
  - `dynamic_analysis.py` (Dynamic harness).
  - `train_model.py` (XGBoost + SHAP pipeline).
- `dataset/` — The generated ELF binaries and their corresponding CSV feature dumps.
- `results/` — The trained `crypto_detector_xgboost.json` model, SHAP plots, ROC curves, and feature rankings.
- `paper/` — The LaTeX source code for the IEEE-formatted academic research article detailing this methodology.

---

## 📖 Research Paper
The academic methodology, dataset synthesis, and deep SHAP interpretability analysis have been rigorously documented. Compile the `paper/paper.tex` file using `pdflatex` to generate the full IEEE-formatted research article.
