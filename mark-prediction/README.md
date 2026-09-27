# 🎓 Multiple Linear Regression - Student Marks Predictor

A Machine Learning project built with **Python**, **pandas**, **numpy**, **matplotlib**, **seaborn**, and **scikit-learn** that predicts a student's final exam score based on **four key input features** using **Multiple Linear Regression**.

---

## 🌟 Key Features & Improvements

- 📊 **100-Row Realistic Sample Dataset:** Contains 100 sample student records labeled across four features:
  1. `Hours Studied` (1.0 to 10.0 hrs/day)
  2. `Sleep Hours` (4.0 to 9.5 hrs/day)
  3. `Attendance Percentage` (55% to 100%)
  4. `Previous Marks` (40 to 98)
  5. `Marks Scored` (Target variable, 10 to 100)
- 🤖 **Model Comparison (Simple vs. Multiple Linear Regression):** Demonstrates significant performance gains when upgrading from 1 feature to 4 features.
- 🏆 **Feature Importance Analysis:** Standardized beta coefficient evaluation to accurately determine **which feature affects marks the most**.
- 🖥️ **Interactive Streamlit App (`app.py`):** Real-time prediction sliders for all four features, model comparison cards, LaTeX equation rendering, and feature impact charts.
- 💻 **CLI & Web API (`predict.py`, `server.py`, `index.html`):** Multi-feature command line predictor and Glassmorphism web application with Chart.js visualization.

---

## 📊 Model Comparison & Metrics

Evaluated on an 80/20 train/test split (80 train samples, 20 test samples):

| Metric | Simple LR (1 Feature: Hours) | Multiple LR (4 Features) | Improvement / Reduction |
| :--- | :---: | :---: | :---: |
| **$R^2$ Variance Score** | `0.7159` (71.6%) | **`0.9465` (94.7%)** | **+23.06% variance explained** |
| **Mean Absolute Error (MAE)** | `6.1345` marks | **`2.5202` marks** | **-58.91% error reduction** |
| **Root Mean Sq Error (RMSE)** | `7.1628` marks | **`3.1071` marks** | **-56.62% error reduction** |

---

## 🏆 Feature Importance Ranking (Which Feature Affects Marks the Most?)

Standardized beta coefficients measure the relative contribution of each feature to the predicted exam score:

| Rank | Feature | Raw Coefficient ($w$) | Standardized Beta | Relative Impact (%) |
| :---: | :--- | :---: | :---: | :---: |
| **1 🥇** | **`Hours Studied`** | **`4.1492`** | **`10.8970`** | **`51.1%` (Most Impactful)** |
| **2 🥈** | **`Previous Marks`** | `0.3411` | `5.5484` | **`26.0%`** |
| **3 🥉** | **`Attendance Percentage`** | `0.2525` | `3.2969` | **`15.4%`** |
| **4** | **`Sleep Hours`** | `1.0099` | `1.5978` | **`7.5%`** |

> **Key Finding:** **Hours Studied** is the single most influential factor, driving **51.1%** of the predicted mark variation, followed by **Previous Marks** (26.0%).

---

## 📐 Multiple Linear Regression Equation

$$\text{Marks Scored} = (4.1492 \times \text{Hours}) + (1.0099 \times \text{Sleep}) + (0.2525 \times \text{Attendance}) + (0.3411 \times \text{PrevMarks}) - 16.2488$$

---

## 🚀 How to Run

### 1. Train Model & Generate Figures
```bash
python train_model.py
```

### 2. Run CLI Predictor
Pass all 4 inputs: `[Hours] [Sleep] [Attendance] [PreviousMarks]`
```bash
python predict.py 8.5 7.5 90.0 85.0
```

### 3. Launch Streamlit Web App
```bash
streamlit run app.py
```

### 4. Launch Flask API & Web Interface
```bash
python server.py
```
*(Open http://localhost:5000 in your web browser)*
