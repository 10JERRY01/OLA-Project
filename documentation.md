# Ola Driver Attrition Prediction - Project Documentation

## 1. Introduction

### 1.1. Problem Statement
Ola faces significant challenges with driver recruitment and retention. High churn rates among drivers lead to increased acquisition costs, operational instability, and potential impacts on organizational morale. This project aims to leverage data science techniques to predict which drivers are likely to leave the company (attrition) based on historical data.

### 1.2. Objective
The primary goal is to build and evaluate machine learning models that can predict driver churn using demographic, tenure, and performance data provided for a segment of drivers from 2019-2020. The insights derived from the model and feature analysis should inform actionable strategies for driver retention.

### 1.3. Dataset
The analysis uses the `ola_driver_scaler.csv` dataset, containing monthly records per driver.

**Column Profiling:**
- `MMM-YY`: Reporting Date (Monthly)
- `Driver_ID`: Unique id for drivers
- `Age`: Age of the driver
- `Gender`: Gender of the driver (0: Male, 1: Female)
- `City`: City Code of the driver
- `Education_Level`: Education level (0: 10+, 1: 12+, 2: Graduate)
- `Income`: Monthly average Income of the driver
- `Dateofjoining`: Joining date for the driver
- `LastWorkingDate`: Last date of working for the driver (NaN if still active)
- `Joining Designation`: Designation of the driver at the time of joining
- `Grade`: Grade of the driver at the time of reporting
- `Total Business Value`: Total business value acquired by the driver in a month
- `Quarterly Rating`: Quarterly rating of the driver (1-5, higher is better)

## 2. Methodology & Analysis Pipeline (`ola_analysis.py`)

The analysis follows a standard data science workflow implemented in the `ola_analysis.py` script.

### 2.1. Data Loading & Initial Checks
- The `ola_driver_scaler.csv` dataset is loaded using pandas.
- The `Unnamed: 0` column (likely a residual index) is dropped.
- Initial exploration includes checking the dataset's shape (`.shape`), data types and non-null counts (`.info()`), and statistical summaries for numerical (`.describe()`) and categorical (`.describe(include='object')`) features.

### 2.2. Data Cleaning & Preprocessing
- **Datetime Conversion:** Columns `MMM-YY`, `Dateofjoining`, and `LastWorkingDate` are converted to datetime objects using `pd.to_datetime`. Error handling and format guessing (`%d/%m/%y`, `dayfirst=False`) are included.
- **Missing Value Assessment:** The number of missing values per column is checked using `df.isnull().sum()`. Significant missingness is observed in `LastWorkingDate` (expected, as it indicates active drivers) and minor missingness in `Age` and `Gender`.

### 2.3. KNN Imputation
- Missing values in the numerical `Age` column are imputed using `sklearn.impute.KNNImputer` with default `n_neighbors=5`. This method uses the values of neighboring data points (based on other numerical features) to estimate the missing age. *Note: Missing `Gender` values were not imputed in this iteration but could be handled similarly or via mode imputation if deemed necessary.*

### 2.4. Data Aggregation
- Since the data contains multiple monthly records per driver, it's aggregated to create a single row representing each unique `Driver_ID`.
- The data is first sorted by `Driver_ID` and `MMM-YY` to ensure consistent selection of 'last' values.
- Aggregation logic:
    - Static features (`Age`, `Gender`, `City`, `Education_Level`, `Dateofjoining`, `Joining Designation`, `Grade`): Take the `'last'` recorded value.
    - Performance/Income (`Income`, `Total Business Value`): Calculate both `'mean'` and `'last'` values.
    - `Quarterly Rating`: Keep `'first'` and `'last'` values for later comparison.
    - Dates (`MMM-YY`, `LastWorkingDate`): Take the `'max'` value (latest reporting month, latest working date).
- MultiIndex columns created during aggregation (e.g., `('Income', 'mean')`) are flattened to single names (e.g., `Income_mean`).
- Column names ending in `_last` or `_max` are cleaned by removing the suffix for better readability (e.g., `Age_last` becomes `Age`).

### 2.5. Feature Engineering
- **`target` Variable:** Created based on the `LastWorkingDate` column. If `LastWorkingDate` is not null, the driver has left (`target = 1`), otherwise `target = 0`.
- **`rating_increase`:** Calculated as 1 if the `Quarterly_Rating` (last) is greater than `Quarterly_Rating_first`, 0 otherwise. The `Quarterly_Rating_first` column is then dropped.
- **`income_increase_over_mean`:** Calculated as 1 if the `Income` (last) is greater than `Income_mean`, 0 otherwise.
- **`tenure_days`:** Calculated as the difference in days between `LastWorkingDate` and `Dateofjoining` for drivers who left, or between `MMM-YY` (latest reporting month) and `Dateofjoining` for active drivers. Negative values are corrected to 0. The original date columns (`Dateofjoining`, `LastWorkingDate`, `MMM-YY`) are dropped after calculating tenure.

### 2.6. Exploratory Data Analysis (Aggregated Data)
- A statistical summary (`.describe(include='all')`) is generated for the final aggregated dataset.
- A correlation matrix is calculated for numerical features using `.corr()`.
- The correlation matrix is visualized using `seaborn.heatmap` and saved as `correlation_heatmap.png`.
- The distribution of the `target` variable is checked to confirm class imbalance.

### 2.7. Encoding Categorical Variables
- Categorical features (identified as `object` type, primarily `City` in this case) are converted into numerical format using `pd.get_dummies` with `drop_first=True` to avoid multicollinearity.
- The script ensures the newly created dummy columns are of integer type.

### 2.8. Data Splitting
- The dataset is split into features (X) and the target variable (y). `Driver_ID` and `target` are excluded from X.
- The data is split into training (80%) and testing (20%) sets using `train_test_split`. `stratify=y` is used to maintain the original class distribution in both sets.

### 2.9. Class Imbalance Handling
- The Synthetic Minority Over-sampling Technique (SMOTE) from the `imblearn` library is applied **only to the training data** (`X_train`, `y_train`) to address the class imbalance observed in the `target` variable. This creates synthetic samples of the minority class (likely `target=0`) to balance the dataset for model training.

### 2.10. Standardization
- Numerical features are standardized using `sklearn.preprocessing.StandardScaler`.
- The scaler is `fit` **only on the resampled training data** (`X_train_resampled`) and then used to `transform` both the training and testing sets (`X_train_resampled`, `X_test`). This prevents data leakage from the test set into the scaling process.

### 2.11. Model Building
Two ensemble models are trained on the scaled, resampled training data:
- **Random Forest (Bagging):** `RandomForestClassifier` with `n_estimators=100` and `class_weight='balanced'`.
- **Gradient Boosting (Boosting):** `GradientBoostingClassifier` with `n_estimators=100`.
*(Note: Hyperparameter tuning using GridSearchCV is commented out but can be enabled for potentially better performance at the cost of longer training time).*

### 2.12. Model Evaluation
- Both models are used to predict the target variable on the **original, scaled test set** (`X_test_scaled`).
- Evaluation metrics are calculated for both models using the test set predictions (`y_pred_rf`, `y_pred_gb`) and true labels (`y_test`):
    - **Classification Report:** Provides precision, recall, and F1-score for each class.
    - **Confusion Matrix:** Shows true positives, true negatives, false positives, and false negatives.
    - **ROC AUC Score:** Calculates the Area Under the Receiver Operating Characteristic curve using predicted probabilities (`y_prob_rf`, `y_prob_gb`).
- The ROC curves for both models are plotted using `matplotlib` and saved as `roc_curve.png`.

### 2.13. Feature Importance
- Feature importances are extracted from both the trained Random Forest (`rf_clf.feature_importances_`) and Gradient Boosting (`gb_clf.feature_importances_`) models.
- The top 10 most important features for each model are printed.

## 3. Results

- **Random Forest:**
    - Accuracy: ~95%
    - ROC AUC: ~0.9770
- **Gradient Boosting:**
    - Accuracy: ~95%
    - ROC AUC: ~0.9811

Both models demonstrate excellent predictive performance on the unseen test data, with Gradient Boosting showing a marginal advantage in ROC AUC score.

## 4. Key Findings & Feature Importance

The feature importance analysis revealed the following key drivers of attrition:

1.  **`tenure_days`:** Overwhelmingly the most important feature for both models. This indicates that the length of time a driver has been with Ola is a critical factor in their decision to stay or leave.
2.  **`Total Business Value`:** The business value generated in the last recorded month is the second most important feature. Lower business value strongly correlates with attrition.
3.  **`Total Business Value_mean`:** The average business value over the driver's recorded history also holds significant predictive power.
4.  **`Quarterly Rating`:** The driver's most recent performance rating is an important factor.
5.  **Other Factors:** `Age`, `Income` (mean and last), and `Gender` show some importance but are less dominant than tenure and business value.

## 5. Actionable Insights & Recommendations

Based on the analysis, the following recommendations can be made to potentially reduce driver churn:

1.  **Tenure-Based Engagement:** Develop programs specifically targeting drivers at different tenure milestones. Early-stage drivers might need more support, while long-term drivers could be rewarded for loyalty. Analyze churn patterns across different tenure lengths to identify critical periods.
2.  **Performance Monitoring & Support:** Implement systems to closely monitor `Total Business Value` and `Quarterly Rating`. Proactively reach out to drivers whose performance metrics decline, offering support, retraining, or investigating underlying issues (e.g., vehicle problems, route difficulties, high cancellations).
3.  **Investigate Low/Negative Business Value:** Conduct deeper analysis into the reasons behind low or negative `Total Business Value`. Are specific routes, times, or customer segments associated with this? Are there issues with the cancellation/refund policy or vehicle EMI deductions impacting drivers negatively?
4.  **Income Stability:** While less dominant than tenure/performance in this model, monitor income trends. Ensure fair compensation and investigate significant drops in earnings that might lead to dissatisfaction.
5.  **Model Deployment:** The high accuracy of the models suggests they can be valuable tools for a proactive retention program. Regularly run the model on current driver data to identify individuals with a high probability of churning and prioritize them for intervention.

## 6. Conclusion

This project successfully developed high-performing machine learning models (Random Forest and Gradient Boosting) capable of predicting Ola driver attrition with approximately 98% ROC AUC accuracy. The analysis identified driver tenure and recent business performance as the most critical factors influencing churn. The generated insights provide a strong basis for Ola to develop targeted and data-driven retention strategies, potentially reducing costly driver turnover and improving operational stability.
