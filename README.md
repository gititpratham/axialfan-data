# 🌀 Tube Axial Fan Performance Analysis & Selection Tool

An interactive engineering dashboard and physics-based analytical suite for performance modeling, interpolation, database management, and cross-fan selection of Tube Axial Fans.

---

## 📌 Overview

This application provides an end-to-end engineering platform for analyzing experimental test data of Tube Axial Fans. Built on deterministic physics principles and fan laws, it calculates 30+ derived aerodynamic and electrical parameters, interpolates performance curves across arbitrary blade angles, and recommends optimal fan-motor combinations for given system requirements.

---

## ✨ Key Features

- **🌐 Comprehensive Fan Registry**: Pre-loaded with **12 Tube Axial Fan models** ranging from 12" to 54" duct diameters.
- **⚡ Physics-Based Interpolation Engine**: Uses cubic polynomial physics fits and fan law speed/density scaling to predict performance curves ($FSP$, $FTP$, $BKW$, $\eta$) at any blade angle without black-box ML instability.
- **🗄️ SQLite Database Layer**: Persistent storage (`fans.db`) with full CRUD support to add new fans, edit test rows, and adjust engineering constants in real time.
- **🌐 Cross-Fan Selection Engine**: Evaluates user system requirements (Volume Flow in CMH/CFM & Static Pressure in mm WG) against all registered fans and motor pole speeds (2, 4, 6 pole) to rank and recommend the best fan-motor-angle match.
- **🔄 Flow Unit Conversion**: Seamless toggle between metric **CMH ($m^3/hr$)** and imperial **CFM ($ft^3/min$)**.
- **📊 Interactive Visualizations**: Plotly-powered charts for Fan Static Pressure ($FSP$), Fan Total Pressure ($FTP$), Brake Power ($BKW$), Static/Total Efficiency ($\eta$), 3D performance surfaces, and system resistance overlays.

---

## 🗂️ Supported Fan Models

| Fan Model | Duct Diameter (in / m) | Default CW | Design Speed (RPM) | Motor Eff ($\eta_m$) | Tested Angles |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **12" Tube Axial Fan** | 12.0" (0.3048 m) | 4.0 | 2850 | 89% | 30°, 35° |
| **14" Tube Axial Fan** | 14.0" (0.3556 m) | 4.0 | 2155 | 85% | 25°, 30° *(Synthesized)* |
| **16" Tube Axial Fan** | 16.0" (0.4064 m) | 4.0 | 1460 | 81% | 20°, 25° |
| **18" Tube Axial Fan** | 18.0" (0.4572 m) | 6.6 | 1460 | 81% | 20°, 30°, 35°, 40°, 45° |
| **24" Tube Axial Fan** | 24.0" (0.6096 m) | 12.7 | 978 | 81% | 20°, 30°, 35°, 37°, 40° |
| **27" Tube Axial Fan** | 27.0" (0.6858 m) | 4.0 | 960 | 81% | 20°, 24°, 28°, 32° |
| **30" Tube Axial Fan** | 30.0" (0.7620 m) | 4.0 | 960 | 81% | 15°, 20°, 24°, 28°, 32°, 35° |
| **33" Tube Axial Fan** | 33.0" (0.8382 m) | 40.0 | 960 | 81% | 15°, 20°, 24°, 28°, 30°, 32°, 35° |
| **36" Tube Axial Fan** | 36.0" (0.9144 m) | 4.0 | 960 | 81% | 14°, 18°, 22°, 25°, 28°, 32°, 35°, 40° |
| **41" Tube Axial Fan** | 41.0" (1.0414 m) | 20.0 | 980 | 81% | 14°, 20°, 22°, 23°, 25°, 28°, 35°, 45° |
| **48" Tube Axial Fan** | 48.0" (1.2192 m) | 80.0 | 1460 | 81% | 24°, 28°, 30° |
| **54" Tube Axial Fan** | 54.0" (1.3716 m) | 20.0 | 980 | 81% | 10°, 24°, 28° |

---

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have Python 3.9+ installed.

### 2. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/gititpratham/axialfan-data.git
cd axialfan-data
pip install -r requirements.txt
```

### 3. Launch Dashboard
```bash
streamlit run app.py
```
The application will open automatically in your default browser at `http://localhost:8501`.

---

## 🖥️ Dashboard Modes & Architecture

### 1. ⚙️ Fan Analysis
- **Live Test Data Editor**: View and edit test rows directly in the UI.
- **Derived Engineering Quantities**: Automatically computes density, velocities, pressure components, power factor, air power, brake power, and efficiencies.
- **Performance Curves**: Plot $FSP$ vs $Q$, $FTP$ vs $Q$, $BKW$ vs $Q$, and Efficiency vs $Q$.
- **Custom Interpolation**: Interpolates exact performance at custom blade angles.
- **Operating Point Matcher**: Recommends standard motor poles (2, 4, 6 pole) and calculates actual operating points.

### 2. 🗄️ Database Manager
- **Manage Database**: View, edit, or delete existing fan records.
- **Constants Editor**: Edit duct diameter, discharge coefficient ($C_D$), wattmeter correction factor ($CW$), design RPM, and motor efficiency.
- **Add Custom Fan**: Register brand-new fan models manually or paste raw CSV test data.

### 3. 🌐 Cross-Fan Selection
- Input required flow volume ($CMH$ / $CFM$) and required static pressure ($mm\ WG$).
- Evaluates **all fans** in the database across all standard motor speeds using physics scaling.
- Outputs a ranked list of candidate models with deviation metrics and performance summaries.

---

## 📐 Engineering Calculations Reference

The computation engine (`data.py`) implements standard fan test formulas:

1. **Test Air Density ($WT$)**:
   $$WT = 1.205 \times \frac{B + 0.0737 \times SP}{760} \times \frac{293}{273 + T_s} \quad \text{[kg/m}^3\text{]}$$

2. **Input Power ($M_i$)**:
   $$M_i = \frac{(W_1 + W_2) \times CW}{1000} \quad \text{[kW]}$$

3. **Test Volume Flow ($Q_t$)**:
   $$Q_t = 12500 \times C_D \times D^2 \times \sqrt{\frac{\Delta P}{WT}} \quad \text{[m}^3\text{/hr]}$$

4. **Speed & Density Scaling (Fan Laws)**:
   $$Q = Q_t \times \left(\frac{N_{design}}{N_{test}}\right)$$
   $$FTP = FTP_t \times \left(\frac{N_{design}}{N_{test}}\right)^2 \times \left(\frac{WT_d}{WT}\right)$$
   $$BKW = M_o \times \left(\frac{N_{design}}{N_{test}}\right)^3 \times \left(\frac{WT_d}{WT}\right)$$

5. **Static & Total Air Power**:
   $$\text{Air Power}_{\text{Total}} = 2.725 \times 10^{-6} \times Q \times FTP \quad \text{[kW]}$$
   $$\eta_{\text{Total}} = \left(\frac{\text{Air Power}_{\text{Total}}}{BKW}\right) \times 100\%$$

---

## 📁 Repository Structure

```
axialfan-data/
├── app.py              # Main Streamlit application entry point & layout
├── app_extensions.py   # Database Manager & Cross-Fan Selection pages
├── data.py             # Fan registry, raw datasets & calculation sheet engine
├── fan_db.py           # SQLite database abstraction layer & persistent storage
├── physics_model.py    # Physics interpolation & cross-fan recommendation engine
├── plots.py            # Plotly interactive curve generation library
├── requirements.txt    # Python package requirements
├── fan_data/           # Persistent database folder
│   └── fans.db         # SQLite database file
└── README.md           # Project documentation
```

