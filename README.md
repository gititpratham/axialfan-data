# 🌀 Axial Fan Performance Analysis Tool (18" & 24")

Interactive Streamlit dashboard for analyzing, predicting, and selecting tube axial fan performance using machine learning. Supports multiple fan models (18" and 24") with dynamic data editing and motor selection recommendations.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
```

The app will open at **http://localhost:8501**.

## Key Features

| Feature | Description |
|---------|-------------|
| **Multi-Fan Support** | Integrated support for 18" and 24" Tube Axial fans with model-specific engineering constants. |
| **Interactive Data Editor** | Edit raw test data directly in the UI, add/delete rows, and retrain ML models in real-time. |
| **ML-Powered Predictions** | Predicts full performance curves (Volume, Pressure, Power, Efficiency) for *any* custom blade angle. |
| **Motor Recommendation** | Input requirements (CMH & SP) → recommends the best motor (950 / 1440 / 2850 RPM) and blade angle using fan laws. |
| **Advanced Visualization** | 10+ interactive Plotly charts including 3D surfaces, system resistance overlays, and predicted vs actual plots. |

## Project Structure

```
fan_performance_app/
├── app.py              # Streamlit dashboard & UI logic
├── data.py             # Fan registry, raw data & calculation engine
├── ml_model.py         # ML pipeline (GBR, RF, GPR) & recommendation logic
├── plots.py            # Plotly visualization library
├── requirements.txt    # Python dependencies
├── .streamlit/         # Dashboard theme configuration
└── README.md           # Documentation
```

## Data Management

The tool uses a **Fan Registry** system allowing for:
- Different duct diameters and design RPMs per fan size.
- Live editing of test data via `st.data_editor`.
- Automatic recalculation of all derived engineering quantities (FSP, FTP, BKW, Efficiency, etc.).

## ML Pipeline

Models are trained on-the-fly using **Leave-One-Out Cross-Validation (LOOCV)** to ensure accuracy on small test datasets.
- **Models:** Gradient Boosting, Random Forest, Gaussian Process.
- **Selection:** The tool automatically selects the model with the highest cross-validated R² score.
- **Extrapolation:** Performance scaling via Fan Laws allows for predictions across standard motor speeds (950, 1440, 2850 RPM).
