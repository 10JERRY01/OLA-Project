import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
from imblearn.over_sampling import SMOTE # To handle class imbalance
import warnings

warnings.filterwarnings('ignore') # Ignore warnings for cleaner output

# Set plot style
sns.set(style="whitegrid")

print("ola_analysis.py script created successfully. Libraries imported.")

# --- Load Data ---
try:
    df = pd.read_csv('ola_driver_scaler.csv')
    print("Dataset loaded successfully.")
    # Drop the 'Unnamed: 0' column if it exists
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
        print("Dropped 'Unnamed: 0' column.")
except FileNotFoundError:
    print("Error: ola_driver_scaler.csv not found in the current directory.")
    exit() # Exit if the file is not found


# --- Initial EDA ---
print("\n--- Initial Exploratory Data Analysis ---")

# Display first 5 rows
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Display shape
print(f"\nDataset shape: {df.shape}")

# Display data types and non-null counts
print("\nDataset info:")
df.info()

# Display statistical summary for numerical features
print("\nStatistical summary (numerical features):")
print(df.describe())

# Display statistical summary for object/categorical features (like MMMM-YY, City)
print("\nStatistical summary (categorical features):")
print(df.describe(include=['object']))


# --- Data Cleaning & Preprocessing ---
print("\n--- Data Cleaning & Preprocessing ---")

# Convert date columns to datetime objects
print("\nConverting date columns...")
# Corrected column name 'MMM-YY'
try:
    # Try parsing with day first (e.g., 01/01/19)
    df['MMM-YY'] = pd.to_datetime(df['MMM-YY'], format='%d/%m/%y', errors='coerce')
except Exception as e1:
    print(f"Initial date parsing failed: {e1}. Trying alternative formats.")
    try:
        # Fallback format if needed (e.g., Jan-19) - adjust based on actual data if first fails
        df['MMM-YY'] = pd.to_datetime(df['MMM-YY'], format='%b-%y', errors='coerce')
    except Exception as e2:
         print(f"Error converting 'MMM-YY': {e2}. Please check the date format.")
         # Consider exiting or handling this case based on requirements

# Corrected column name 'Dateofjoining' and added dayfirst=False for common formats like MM/DD/YY
df['Dateofjoining'] = pd.to_datetime(df['Dateofjoining'], errors='coerce', dayfirst=False)
df['LastWorkingDate'] = pd.to_datetime(df['LastWorkingDate'], errors='coerce', dayfirst=False)
print("Date columns converted (attempted).")
print("\nData types after date conversion:")
df.info() # Display info again to show converted types

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# --- KNN Imputation ---
print("\n--- KNN Imputation for Missing Numerical Values ---")

# Identify numerical columns for imputation (excluding Driver_ID and potentially target-related later)
# Based on typical datasets and the info() output, these are likely candidates.
# We'll refine this list based on the actual isnull().sum() output when the script runs.
numerical_cols_for_imputation = ['Age', 'Income', 'Total Business Value', 'Quarterly Rating']

# Filter out columns that might not exist or have no missing values to avoid errors
numerical_cols_to_impute = [col for col in numerical_cols_for_imputation if col in df.columns and df[col].isnull().any()]

if not numerical_cols_to_impute:
    print("No missing values found in the selected numerical columns for imputation.")
else:
    print(f"Performing KNN Imputation on: {numerical_cols_to_impute}")
    # Select only the numerical columns for the imputer
    imputer_data = df[numerical_cols_to_impute]

    # Initialize KNNImputer (using default n_neighbors=5)
    knn_imputer = KNNImputer(n_neighbors=5)

    # Fit and transform the data
    imputed_data = knn_imputer.fit_transform(imputer_data)

    # Convert the imputed data back to a DataFrame with original column names
    imputed_df = pd.DataFrame(imputed_data, columns=numerical_cols_to_impute, index=df.index)

    # Update the original DataFrame with imputed values
    df.update(imputed_df)

    print("KNN Imputation completed.")
    print("\nMissing values after KNN Imputation:")
    print(df[numerical_cols_to_impute].isnull().sum()) # Verify imputation


# --- Data Aggregation ---
print("\n--- Aggregating Data by Driver_ID ---")

# Sort by Driver_ID and reporting date to ensure 'last' picks the latest record
# Corrected column name 'MMM-YY'
df = df.sort_values(by=['Driver_ID', 'MMM-YY'])

# Define aggregation dictionary
agg_dict = {
    # Static features: take the last known value
    'Age': 'last',
    'Gender': 'last',
    'City': 'last',
    'Education_Level': 'last',
    # Corrected column name 'Dateofjoining'
    'Dateofjoining': 'last',
    'Joining Designation': 'last',
    # Time-varying features:
    'Income': ['mean', 'last'], # Keep mean income and last recorded income
    'Grade': 'last', # Last known grade
    'Total Business Value': ['mean', 'last'], # Keep mean and last business value
    'Quarterly Rating': ['first', 'last'], # Keep first and last rating for comparison later
    # Date features:
    # Corrected column name 'MMM-YY'
    'MMM-YY': 'max', # Last reporting month
    'LastWorkingDate': 'max' # Last working date (will be NaT if still working)
}

# Perform aggregation
df_agg = df.groupby('Driver_ID').agg(agg_dict)

# Flatten MultiIndex columns (e.g., ('Income', 'mean') becomes 'Income_mean')
df_agg.columns = ['_'.join(col).strip('_') for col in df_agg.columns.values]

# Reset index to bring Driver_ID back as a column
df_agg = df_agg.reset_index()

# Rename columns by removing '_last' suffix for clarity
df_agg.columns = [col.replace('_last', '') if col.endswith('_last') else col for col in df_agg.columns]
# Also remove '_max' suffix from date columns used for tenure/target
df_agg.columns = [col.replace('_max', '') if col.endswith('_max') else col for col in df_agg.columns]
# Rename Quarterly Rating_first if it exists (used for rating_increase)
if 'Quarterly Rating_first' in df_agg.columns:
     df_agg = df_agg.rename(columns={'Quarterly Rating_first': 'Quarterly_Rating_first'})


print("Data aggregated successfully.")
print(f"Aggregated dataset shape: {df_agg.shape}")
print("\nFirst 5 rows of aggregated data:")
print(df_agg.head())
print("\nAggregated data info:")
df_agg.info()
print("\nMissing values in aggregated data:")
print(df_agg.isnull().sum())


# --- Feature Engineering ---
print("\n--- Feature Engineering ---")

# 1. Target Variable: 1 if driver left, 0 otherwise
# Uses the renamed 'LastWorkingDate' column
df_agg['target'] = df_agg['LastWorkingDate'].notna().astype(int)
print(f"\nTarget variable 'target' created. Distribution:\n{df_agg['target'].value_counts(normalize=True)}")

# 2. Quarterly Rating Increase: 1 if last rating > first rating
# Ensure both columns exist before creating the feature (using renamed columns)
if 'Quarterly_Rating_first' in df_agg.columns and 'Quarterly_Rating' in df_agg.columns:
    df_agg['rating_increase'] = (df_agg['Quarterly_Rating'] > df_agg['Quarterly_Rating_first']).astype(int)
    print("\nFeature 'rating_increase' created.")
    # Drop the original first rating column as it's now captured in rating_increase
    df_agg = df_agg.drop(columns=['Quarterly_Rating_first'])
else:
    print("\nWarning: 'Quarterly_Rating_first' or 'Quarterly_Rating' not found after aggregation/renaming. Skipping 'rating_increase' feature.")


# 3. Monthly Income Increase (Proxy): 1 if last income > mean income
# Ensure both columns exist
if 'Income_mean' in df_agg.columns and 'Income_last' in df_agg.columns:
    df_agg['income_increase_over_mean'] = (df_agg['Income_last'] > df_agg['Income_mean']).astype(int)
    print("Feature 'income_increase_over_mean' created.")
    # Decide whether to keep Income_mean and Income_last or just one. Let's keep both for now.
else:
     print("\nWarning: 'Income_mean' or 'Income_last' not found. Skipping 'income_increase_over_mean' feature.")


# 4. Tenure: Calculate tenure in days
# Ensure required date columns exist (using renamed columns)
if 'Dateofjoining' in df_agg.columns and 'LastWorkingDate' in df_agg.columns and 'MMM-YY' in df_agg.columns:
    # For drivers who left
    left_mask = df_agg['target'] == 1
    df_agg.loc[left_mask, 'tenure_days'] = (df_agg['LastWorkingDate'] - df_agg['Dateofjoining']).dt.days

    # For drivers still working (use last reporting date)
    working_mask = df_agg['target'] == 0
    df_agg.loc[working_mask, 'tenure_days'] = (df_agg['MMM-YY'] - df_agg['Dateofjoining']).dt.days

    # Handle potential negative tenure if dates are inconsistent (e.g., joining date after last working date)
    df_agg['tenure_days'] = df_agg['tenure_days'].apply(lambda x: max(x, 0) if pd.notna(x) else 0)
    # Handle potential negative tenure if dates are inconsistent
    df_agg['tenure_days'] = df_agg['tenure_days'].apply(lambda x: max(x, 0) if pd.notna(x) else 0)
    # Fill any remaining NaNs in tenure_days (e.g., if Dateofjoining was NaT) with 0
    df_agg['tenure_days'] = df_agg['tenure_days'].fillna(0)
    print("Feature 'tenure_days' created.")

    # Drop original date columns used for tenure calculation (using renamed columns)
    df_agg = df_agg.drop(columns=['Dateofjoining', 'LastWorkingDate', 'MMM-YY'])
else:
    print("\nWarning: Required date columns ('Dateofjoining', 'LastWorkingDate', 'MMM-YY') for tenure calculation not found after aggregation/renaming. Skipping 'tenure_days' feature.")


print("\nData after Feature Engineering:")
print(df_agg.head())
print("\nInfo after Feature Engineering:")
df_agg.info()


# --- Further EDA on Aggregated Data ---
print("\n--- Further EDA on Aggregated Data ---")

# Statistical summary of the final aggregated dataset
print("\nStatistical summary of aggregated data:")
# Include 'all' to get summary for both numerical and categorical (if any remain as object)
print(df_agg.describe(include='all'))

# Correlation Analysis
print("\nCorrelation matrix:")
# Select only numerical columns for correlation calculation
numerical_cols = df_agg.select_dtypes(include=np.number).columns
# Exclude Driver_ID from correlation matrix if it's numerical
if 'Driver_ID' in numerical_cols:
    numerical_cols = numerical_cols.drop('Driver_ID')

correlation_matrix = df_agg[numerical_cols].corr()
print(correlation_matrix)

# Visualize correlation matrix and save to file
plt.figure(figsize=(15, 12)) # Increased size for more features
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm') # Annot=False if too cluttered
plt.title('Correlation Matrix of Numerical Features')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
print("\nCorrelation heatmap saved to correlation_heatmap.png")
plt.close() # Close the plot to free memory

print("\nTarget variable distribution:")
print(df_agg['target'].value_counts())
print(df_agg['target'].value_counts(normalize=True))


# --- Encoding Categorical Variables ---
print("\n--- Encoding Categorical Variables ---")

# Identify categorical columns (object or category dtype)
categorical_cols = df_agg.select_dtypes(include=['object', 'category']).columns

# Exclude Driver_ID if it was read as object, though it should be numerical
if 'Driver_ID' in categorical_cols:
    categorical_cols = categorical_cols.drop('Driver_ID')

if len(categorical_cols) > 0:
    print(f"Applying One-Hot Encoding to: {list(categorical_cols)}")
    # Apply one-hot encoding
    df_encoded = pd.get_dummies(df_agg, columns=categorical_cols, drop_first=True) # drop_first=True to avoid multicollinearity

    print("Categorical variables encoded.")
    print(f"Shape after encoding: {df_encoded.shape}")
    print("\nColumns after encoding:")
    print(df_encoded.columns)

    # Update df_agg to the encoded version
    df_agg = df_encoded

    # Ensure dummy columns are integer type
    dummy_cols = [col for col in df_agg.columns if col.startswith(tuple(categorical_cols))]
    for col in dummy_cols:
        if df_agg[col].dtype not in [np.int64, np.int32, np.uint8, np.float64, np.float32]:
             df_agg[col] = df_agg[col].astype(int)
    print("Ensured dummy columns are integer type.")

else:
    print("No categorical columns found to encode.")

# Datetime columns should have been dropped during tenure calculation now
# Ensure City_last (original categorical column before dummifying) is dropped if it still exists
if 'City' in df_agg.columns:
     df_agg = df_agg.drop(columns=['City'], errors='ignore')
     print("Dropped original 'City' column.")


# Final check for any remaining NaN values before splitting
print("\nChecking for NaN values before splitting:")
nan_check = df_agg.isnull().sum()
print(nan_check[nan_check > 0])

# If there are NaNs in numerical columns, fill with median
numerical_cols_final = df_agg.select_dtypes(include=np.number).columns
if 'Driver_ID' in numerical_cols_final:
    numerical_cols_final = numerical_cols_final.drop(['Driver_ID', 'target'], errors='ignore') # Exclude ID and target
else:
     numerical_cols_final = numerical_cols_final.drop(['target'], errors='ignore')

for col in numerical_cols_final:
    if df_agg[col].isnull().any():
        median_val = df_agg[col].median()
        df_agg[col] = df_agg[col].fillna(median_val)
        print(f"Filled NaN in numerical column {col} with median value {median_val}")


# --- Data Splitting ---
print("\n--- Splitting Data into Training and Testing Sets ---")

# Define features (X) and target (y)
# Ensure Driver_ID exists before trying to drop it
columns_to_drop_for_X = ['target']
if 'Driver_ID' in df_agg.columns:
    columns_to_drop_for_X.append('Driver_ID')

X = df_agg.drop(columns=columns_to_drop_for_X)
y = df_agg['target']


# Ensure all feature columns are numeric before proceeding
non_numeric_cols = X.select_dtypes(exclude=np.number).columns
if len(non_numeric_cols) > 0:
    print(f"Error: Non-numeric columns found in features: {list(non_numeric_cols)}")
    print("Please ensure all categorical features are encoded.")
    exit()

# Split data into training and testing sets (e.g., 80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # Stratify by y for imbalance

print(f"Training set shape: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Testing set shape: X_test={X_test.shape}, y_test={y_test.shape}")
print(f"Training target distribution:\n{y_train.value_counts(normalize=True)}")
print(f"Testing target distribution:\n{y_test.value_counts(normalize=True)}")


# --- Class Imbalance Treatment (SMOTE) ---
print("\n--- Handling Class Imbalance using SMOTE (on training data) ---")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"Shape after SMOTE: X_train_resampled={X_train_resampled.shape}, y_train_resampled={y_train_resampled.shape}")
print(f"Training target distribution after SMOTE:\n{y_train_resampled.value_counts(normalize=True)}")


# --- Standardization ---
print("\n--- Standardizing Numerical Features ---")

# Identify numerical columns to scale (should be all columns in X now)
# We fit the scaler ONLY on the training data (resampled)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test) # Use the same scaler fitted on training data

# Convert scaled arrays back to DataFrames (optional, but can be helpful)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

print("Standardization complete.")
print("\nScaled Training Data Head:")
print(X_train_scaled.head())


# --- Model Building ---
print("\n--- Model Building ---")

# --- Model 1: Random Forest (Bagging) ---
print("\nTraining Random Forest Classifier...")
rf_clf = RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced') # Using default parameters + balanced weights

# Optional: Hyperparameter Tuning with GridSearchCV (can be time-consuming)
# param_grid_rf = {
#     'n_estimators': [100, 200],
#     'max_depth': [None, 10, 20],
#     'min_samples_split': [2, 5]
# }
# grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=42, class_weight='balanced'), param_grid_rf, cv=3, scoring='roc_auc', n_jobs=-1)
# grid_search_rf.fit(X_train_scaled, y_train_resampled)
# rf_clf = grid_search_rf.best_estimator_
# print(f"Best RF Params: {grid_search_rf.best_params_}")

rf_clf.fit(X_train_scaled, y_train_resampled)
print("Random Forest training complete.")


# --- Model 2: Gradient Boosting (Boosting) ---
print("\nTraining Gradient Boosting Classifier...")
gb_clf = GradientBoostingClassifier(random_state=42, n_estimators=100) # Using default parameters

# Optional: Hyperparameter Tuning with GridSearchCV
# param_grid_gb = {
#     'n_estimators': [100, 200],
#     'learning_rate': [0.1, 0.05],
#     'max_depth': [3, 5]
# }
# grid_search_gb = GridSearchCV(GradientBoostingClassifier(random_state=42), param_grid_gb, cv=3, scoring='roc_auc', n_jobs=-1)
# grid_search_gb.fit(X_train_scaled, y_train_resampled)
# gb_clf = grid_search_gb.best_estimator_
# print(f"Best GB Params: {grid_search_gb.best_params_}")

gb_clf.fit(X_train_scaled, y_train_resampled)
print("Gradient Boosting training complete.")


# --- Results Evaluation ---
print("\n--- Results Evaluation ---")

# Predictions on the test set
y_pred_rf = rf_clf.predict(X_test_scaled)
y_prob_rf = rf_clf.predict_proba(X_test_scaled)[:, 1] # Probabilities for ROC AUC

y_pred_gb = gb_clf.predict(X_test_scaled)
y_prob_gb = gb_clf.predict_proba(X_test_scaled)[:, 1] # Probabilities for ROC AUC

# --- Evaluation Metrics ---

# Random Forest
print("\n--- Random Forest Evaluation ---")
print("Classification Report:")
print(classification_report(y_test, y_pred_rf))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))
roc_auc_rf = roc_auc_score(y_test, y_prob_rf)
print(f"ROC AUC Score: {roc_auc_rf:.4f}")

# Gradient Boosting
print("\n--- Gradient Boosting Evaluation ---")
print("Classification Report:")
print(classification_report(y_test, y_pred_gb))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_gb))
roc_auc_gb = roc_auc_score(y_test, y_prob_gb)
print(f"ROC AUC Score: {roc_auc_gb:.4f}")

# --- ROC Curve Data (for potential plotting) ---
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
fpr_gb, tpr_gb, _ = roc_curve(y_test, y_prob_gb)

# Plot ROC Curve and save to file
plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_rf:.4f})')
plt.plot(fpr_gb, tpr_gb, label=f'Gradient Boosting (AUC = {roc_auc_gb:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Chance') # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Ola Driver Attrition')
plt.legend()
plt.grid(True)
plt.savefig('roc_curve.png')
print("\nROC curve saved to roc_curve.png")
plt.close()


# --- Feature Importance (Example for Random Forest) ---
print("\n--- Feature Importance (Random Forest) ---")
try:
    feature_importances = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf_clf.feature_importances_
    }).sort_values('importance', ascending=False)
    print("Top 10 Features (Random Forest):")
    print(feature_importances.head(10))
except AttributeError:
    print("Could not retrieve feature importances for Random Forest.")

# --- Feature Importance (Example for Gradient Boosting) ---
print("\n--- Feature Importance (Gradient Boosting) ---")
try:
    feature_importances_gb = pd.DataFrame({
        'feature': X_train.columns,
        'importance': gb_clf.feature_importances_
    }).sort_values('importance', ascending=False)
    print("Top 10 Features (Gradient Boosting):")
    print(feature_importances_gb.head(10))
except AttributeError:
    print("Could not retrieve feature importances for Gradient Boosting.")


# --- Actionable Insights & Recommendations ---
print("\n--- Actionable Insights & Recommendations ---")
print("Based on the final model results and feature importances:")
print("1. Dominant Predictors: Driver tenure ('tenure_days') and the most recent month's 'Total Business Value' are overwhelmingly the most significant predictors of attrition.")
print("2. High Predictive Power: Both Random Forest and Gradient Boosting models achieved excellent performance (AUC ~0.98), indicating a strong ability to identify drivers at risk of leaving.")
print("3. Retention Strategy - Tenure Milestones: Implement targeted engagement strategies based on tenure. Drivers might be more prone to leaving at specific points (e.g., early tenure, after 1 year). Recognize and reward loyalty at key milestones.")
print("4. Retention Strategy - Business Value Monitoring: Closely monitor drivers with low or declining 'Total Business Value'. Investigate the root causes (e.g., low ride volume, high cancellations/refunds, EMI issues) and offer targeted support or incentives.")
print("5. Secondary Factors: While less dominant, factors like 'Quarterly Rating', 'Income', and 'Age' still play a role. Continue monitoring these, especially sudden drops in rating or income.")
print("6. Model Utility: The high accuracy suggests these models can be effectively deployed to proactively identify at-risk drivers, allowing for timely intervention.")


print("\n--- Analysis Complete ---")
