# Diabetes Progression Prediction with Linear Regression

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/) 
[![scikit-learn](https://img.shields.io/badge/scikit--learn-0.24%2B-f7931e?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/) 
[![seaborn](https://img.shields.io/badge/seaborn-0.11%2B-4c8cbf?logo=seaborn&logoColor=white)](https://seaborn.pydata.org/) 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE) 
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/) 
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-blue.svg)](https://www.kaggle.com/code/evangelosgakias/linear-regression-diabetes) 
[![Reproducible Research](https://img.shields.io/badge/Reproducible-Yes-brightgreen.svg)](https://www.kaggle.com/code/evangelosgakias/linear-regression-diabetes)

---

## 🚀 Live Results

You can view the notebook with all outputs and results on Kaggle:
[https://www.kaggle.com/code/evangelosgakias/linear-regression-diabetes](https://www.kaggle.com/code/evangelosgakias/linear-regression-diabetes)

All metrics, plots, and outputs are available in the linked Kaggle notebook for full transparency and reproducibility.

---

## 📑 Table of Contents
- [Live Results](#-live-results)
- [Table of Contents](#-table-of-contents)
- [Overview](#-overview)
- [Project Structure](#-project-structure)
- [Features](#-features)
- [Quickstart](#-quickstart)
- [Usage](#-usage)
- [Results](#-results)
- [Limitations and Future Work](#-limitations-and-future-work)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 📝 Overview
This project presents a comprehensive machine learning workflow for predicting diabetes progression using **linear regression** on the classic scikit-learn diabetes dataset. The notebook demonstrates:
- End-to-end data science best practices (EDA, feature engineering, modeling, evaluation, and interpretation)
- Both scikit-learn and custom gradient descent implementations
- Professional documentation, accessibility, and reproducibility standards

**Goal:** Predict a quantitative measure of diabetes progression one year after baseline, using 10 physiological features from 442 patients. This project is ideal for those seeking a clear, portfolio-ready example of regression analysis in healthcare data.

---

## 🏗️ Project Structure
```
Linear Regression/
├── LR.ipynb           # Jupyter notebook with the complete implementation
├── README.md          # Project documentation (this file)
├── requirements.txt   # Python dependencies
├── LICENSE            # MIT License file
└── .venv/             # (Optional) Virtual environment directory
```

---

## 🚀 Features

### Data Preparation
- **Dataset Loading**: Uses the scikit-learn diabetes dataset (10 physiological features, 442 patients)
- **Exploratory Data Analysis (EDA)**: Statistical summaries, correlation analysis, and visualizations (histograms, pairplots, heatmaps)
- **Preprocessing**:
  - Feature selection (all 10 baseline variables)
  - Standardization of features
  - Train/test split (80%/20%)

### Modeling
- **Linear Regression (Closed-Form Solution)**: Utilizes scikit-learn's `LinearRegression` for the normal equation approach
- **Custom Gradient Descent Implementation**: Implements linear regression from scratch using gradient descent for educational comparison
- **Pipeline**: Combines scaling and modeling for reproducibility

### Evaluation & Interpretation
- **Metrics**: Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), R² (coefficient of determination)
- **Cross-Validation**: 5-fold cross-validation to assess model stability
- **Baseline Comparison**: Dummy regressor for baseline performance
- **Feature Importance**: Coefficient analysis and bar chart visualization
- **Residual Analysis**: Residual plots and predicted vs. actual scatter plots

*All plots include descriptive titles, axis labels, and are designed for accessibility.*


---

## ⚡ Quickstart
1. **Kaggle (Recommended for Reproducibility):**
   - [Run the notebook on Kaggle](https://www.kaggle.com/code/evangelosgakias/linear-regression-diabetes)
2. **Local:**
   - Clone the repo and run `LR.ipynb` in Jupyter after installing requirements.

---

## 💻 Usage
1. **📥 Clone the repository:**
   ```bash
   git clone https://github.com/EvanGks/linear-regression-diabetes-sklearn.git
   cd linear-regression-diabetes-sklearn
   ```
2. **🔒 Create and activate a virtual environment:**
   - **Windows:**
     ```bash
     python -m venv .venv
     .venv\Scripts\activate
     ```
   - **macOS/Linux:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
3. **📦 Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **🚀 Launch Jupyter Notebook:**
   ```bash
   jupyter notebook LR.ipynb
   ```
5. **▶️ Run all cells** to reproduce the analysis and results.

**🛠️ Troubleshooting:**
- If you encounter missing package errors, ensure your Python environment is activated and up to date.
- For best reproducibility, use the provided Kaggle link.

---

## 📊 Results
### Model Metrics
- **Cross-Validation R²:** Mean ≈ 0.48 (range: 0.39–0.58)
- **MAE:** ~42.5
- **MSE:** ~2900
- **RMSE:** ~53.8
- **Baseline (Dummy Regressor) R²:** ≈ -0.02

### Feature Importance
Top predictors (by absolute coefficient):
1. **bmi** (Body Mass Index)
2. **s5** (Serum measurements)
3. **bp** (Blood Pressure)

### Visualizations
- **Histograms:** Distribution of each feature and target
- **Residual Plots:** Visual check for model fit and outliers
- **Predicted vs. Actual:** Scatter plot to assess prediction quality
- **Feature Importance Bar Chart:** Visual ranking of coefficients

*All plots include descriptive titles, axis labels, and alt text for accessibility.*

---

## 📝 Limitations and Future Work
- **Linear Assumption:** May not capture complex, non-linear relationships
- **Sensitivity to Outliers:** Real-world data may require robust preprocessing
- **Potential Improvements:**
  - Explore non-linear models (e.g., tree-based, neural networks)
  - Apply regularization (Ridge, Lasso)
  - Advanced feature engineering
  - More sophisticated optimization (e.g., Adam, momentum)
  - Deploy as a web app with accessible UI

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

---

## 📝 License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

---

## 📬 Contact
For questions or feedback, please reach out via:

- **GitHub:** [EvanGks](https://github.com/EvanGks)
- **X (Twitter):** [@Evan6471133782](https://x.com/Evan6471133782)
- **LinkedIn:** [Evangelos Gakias](https://www.linkedin.com/in/evangelos-gakias-346a9072)
- **Kaggle:** [evangelosgakias](https://www.kaggle.com/evangelosgakias)
- **Email:** [vgakias_@hotmail.com](mailto:vgakias_@hotmail.com)

---
Happy Coding! 🚀

