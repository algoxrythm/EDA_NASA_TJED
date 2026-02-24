---

# 🚀 NASA Turbofan Engine Degradation – EDA Project

![NASA](https://img.shields.io/badge/Dataset-NASA--CMAPSS-blue)
📊 Exploratory Data Analysis (EDA) • 🧠 Predictive Maintenance • ✈️ Aerospace Sensor Data

---

## 📌 Overview

This project explores the **NASA Turbofan Engine Degradation Simulation Dataset (CMAPSS)** to understand how aircraft engine sensors behave over time and identify patterns that signal **degradation and failure**.

🔍 The goal is to perform **advanced EDA** to uncover:

* Which sensors are important
* How engines degrade over time
* What features may be useful for predicting **Remaining Useful Life (RUL)**

---

## 📂 Dataset Details

**Source**: NASA Ames Prognostics Center of Excellence  
**Data Type**: Multivariate Time-Series  
**Format**: Each row represents a single time step for one engine.

| Column                    | Description                                   |
| ------------------------- | --------------------------------------------- |
| `unit`                    | Engine ID                                     |
| `time_in_cycles`          | Operational cycle (time step)                 |
| `op_setting_*`            | Environmental/operational conditions          |
| `sensor_1` to `sensor_21` | Sensor readings (e.g., temperature, pressure) |

---

## ⚙️ Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## ▶️ Run the analysis

Place `train_FD001.txt` locally (for example in `data/`), then run:

```bash
python eda_nasa_tjed.py --data-path data/train_FD001.txt --output-dir outputs
```

Generated artifacts include:

- `outputs/summary.txt`
- `outputs/failure_cycle_distribution.png`
- `outputs/sensor_distributions.png`
- `outputs/sensor_boxplots.png`
- `outputs/sensor_correlation_matrix.png`
- `outputs/selected_sensors_vs_rul.png`
- `outputs/selected_features.txt`
- `outputs/feature_importances.csv`
- `outputs/top10_feature_importances.png`

---

## 📈 Visualizations

The script now saves all visuals locally to `outputs/` so they render meaningfully in your own runs.  
To publish images on GitHub, commit selected files from `outputs/` into a folder like `docs/images/` and link them here.

---

## 🛠 Tools & Libraries

* Python
* `pandas`, `numpy` – data manipulation
* `matplotlib`, `seaborn` – visualizations
* `scikit-learn` – baseline feature screening/modeling

---

## 🔍 Key Insights (from baseline EDA)

* 📉 **Sensor 2, 3, 11, and 15** often show degradation-relevant trends.
* ⚙️ Some sensors are near-constant and can be candidates for removal.
* 🔁 Engines degrade **non-uniformly**, so robust models must generalize across units.
* 🚦 RUL can be learned from selected sensor patterns.

---

## 🧭 What's Next?

* 🔧 Feature engineering: rolling averages, deltas, lag features
* 🤖 Compare ML models for RUL (Linear, Random Forest, LSTM)
* 💻 Optional deployment as Streamlit dashboard

---

## 📚 References

* [NASA CMAPSS Dataset](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository)
* [Original Paper](https://www.phmsociety.org/node/104)

---
