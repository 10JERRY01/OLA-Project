# Ola Driver Attrition Prediction Project

## Project Overview

This project aims to predict driver attrition for Ola, a major ride-sharing service. High driver churn is a significant challenge in the industry, impacting operational efficiency and incurring high acquisition costs. By analyzing historical driver data from 2019-2020, this project builds machine learning models to identify drivers likely to leave the company.

The analysis involves:
- Exploratory Data Analysis (EDA) to understand driver demographics, tenure, and performance.
- Data preprocessing, including handling missing values (KNN Imputation) and feature engineering (e.g., calculating tenure, identifying rating/income changes).
- Handling class imbalance using SMOTE.
- Building and evaluating ensemble models (Random Forest and Gradient Boosting).
- Deriving actionable insights to inform driver retention strategies.

## Dataset

The primary dataset used is `ola_driver_scaler.csv`. It contains monthly information for a segment of drivers, including:
- Demographics (Age, Gender, City, Education)
- Tenure (Joining Date, Last Working Date)
- Performance (Quarterly Rating, Total Business Value, Grade, Income)

*(Note: The dataset `ola_driver_scaler.csv` is required to run the analysis but is not included in this repository due to its size).*

## Setup and Dependencies

This project uses Python and several data science libraries.

1.  **Clone the repository (or ensure you have the files):**
    ```bash
    # If this were a git repo:
    # git clone <repository_url>
    # cd <repository_directory>
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # Activate the environment (Windows example):
    .\venv\Scripts\activate
    # (macOS/Linux example):
    # source venv/bin/activate
    ```
3.  **Install dependencies:** Ensure you have Python installed. You'll need the following libraries, which can typically be installed via pip:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
    ```

## Running the Analysis

1.  Place the `ola_driver_scaler.csv` dataset in the same directory as the `ola_analysis.py` script.
2.  Run the Python script from your terminal:
    ```bash
    python ola_analysis.py
    ```
3.  The script will execute the complete analysis pipeline, printing logs, evaluation metrics, feature importances, and insights to the console.
4.  It will also generate two image files in the project directory:
    - `correlation_heatmap.png`: Visualizing correlations between numerical features.
    - `roc_curve.png`: Comparing the ROC curves of the Random Forest and Gradient Boosting models.

## Key Findings

- **High Predictive Accuracy:** Both Random Forest and Gradient Boosting models achieved excellent performance (Accuracy ~95%, ROC AUC ~0.98) in predicting driver churn.
- **Dominant Predictors:** Driver tenure (`tenure_days`) and the most recent month's `Total Business Value` are the most significant factors influencing attrition.
- **Other Factors:** `Quarterly Rating`, `Income`, and `Age` also contribute to the prediction.

## Documentation

For a detailed explanation of the project methodology, data processing steps, model building, and in-depth insights, please refer to `documentation.md`.
