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
| `unit_number`             | Engine ID                                     |
| `time_in_cycles`          | Operational cycle (time step)                 |
| `operational_setting_*`   | Environmental/operational conditions          |
| `sensor_1` to `sensor_21` | Sensor readings (e.g., temperature, pressure) |

---

## 🧠 Problem Statement

> **"Can we explore and visualize the patterns in sensor data that indicate engine degradation and estimate Remaining Useful Life (RUL)?"**

---

## 🧪 What I Did

| Task                    | Description                                       |
| ----------------------- | ------------------------------------------------- |
| ✅ Data Loading          | Cleaned and structured the CMAPSS data            |
| 🧮 RUL Calculation      | Computed Remaining Useful Life for each engine    |
| 📊 Sensor Distributions | Plotted histograms, line plots, and boxplots      |
| 📉 Sensor vs RUL        | Analyzed how sensor values degrade as engines age |
| 🔗 Correlation Analysis | Found relationships among sensors                 |
| 🔍 Unit-Level Analysis  | Compared degradation trends across engines        |
| ✨ Insights & Summary    | Highlighted patterns that help in ML modeling     |

---

## 📈 Sample Visualizations

<p align="center">
  <img src="https://user-images.githubusercontent.com/your_image_url_here" width="600" alt="RUL Trends">
  <br><em>Sensor 11 vs RUL across different engines</em>
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/your_image_url_here" width="600" alt="Correlation Matrix">
  <br><em>Sensor Correlation Heatmap</em>
</p>

---

## 🛠 Tools & Libraries

* Python (Google Colab)
* `pandas`, `numpy` – data manipulation
* `matplotlib`, `seaborn`, `plotly` – visualizations
* `scikit-learn` (for future modeling)

---

## 🔍 Key Insights

* 📉 **Sensor 2, 3, 11, and 15** showed clear degradation over time.
* ⚙️ Some sensors were flat or constant — candidates for removal.
* 🔁 Engines degrade **non-uniformly**, meaning robust models must generalize.
* 🚦 RUL is predictable based on selected sensor patterns.

---

## 🧭 What's Next?

* 🔧 Feature engineering: rolling averages, deltas, etc.
* 🤖 Build ML models to predict RUL (Linear, Random Forest, LSTM)
* 💻 Deploy as a live dashboard or Streamlit app

---

## 📚 References

* [NASA CMAPSS Dataset](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository)
* [Original Paper](https://www.phmsociety.org/node/104)

---

## 🙋‍♂️ Author

**Pranav Patil**
3rd Year Robotics & Automation Engineering Student
📌 India | 🔧 ML, Robotics, and Embedded Systems Enthusiast
📫 [LinkedIn](https://www.linkedin.com/in/yourprofile) • [GitHub](https://github.com/yourprofile)

---
