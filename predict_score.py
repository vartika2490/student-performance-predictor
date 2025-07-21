import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration Section: You might edit these if your data changes ---
DATA_FILE_NAME = 'student_data.csv'
# Define your feature columns (the inputs for prediction)
FEATURE_COLUMNS = ['hours_studied', 'previous_score', 'attendance_percentage', 'gender']
# Define your target column (what you want to predict)
TARGET_COLUMN = 'exam_score'

# --- Step 1: Load the Dataset ---
try:
    df = pd.read_csv(DATA_FILE_NAME)
    print("Dataset loaded successfully!")
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
    print("\nDataset Info:")
    df.info()
except FileNotFoundError:
    print(f"Error: '{DATA_FILE_NAME}' not found.")
    print("Please ensure the CSV file is in the same directory as this Python script,")
    print("or provide the full path to the file in DATA_FILE_NAME variable.")
    exit() # Exit if the file isn't found, as we can't proceed without data.

# --- Step 2: Exploratory Data Analysis (EDA) - Optional but Recommended ---
print("\n--- Descriptive Statistics ---")
print(df.describe(include='all')) # include='all' to get stats for categorical too

print("\n--- Missing Values ---")
print(df.isnull().sum())

# Visualize distributions of numerical features and target
# Adjust column names if your actual dataset has different ones
numerical_for_plot = ['hours_studied', 'previous_score', 'attendance_percentage', 'exam_score']
plt.figure(figsize=(18, 5))
for i, col in enumerate(numerical_for_plot):
    plt.subplot(1, len(numerical_for_plot), i + 1)
    sns.histplot(df[col], kde=True)
    plt.title(f'{col} Distribution')
plt.tight_layout()
plt.show()

# Visualize relationships with exam_score
plt.figure(figsize=(18, 5))
features_for_scatter = ['hours_studied', 'previous_score', 'attendance_percentage']
for i, col in enumerate(features_for_scatter):
    plt.subplot(1, len(features_for_scatter), i + 1)
    sns.scatterplot(x=col, y=TARGET_COLUMN, data=df)
    plt.title(f'{col} vs. {TARGET_COLUMN}')
plt.tight_layout()
plt.show()

# Correlation matrix for numerical features
print("\n--- Correlation Matrix (Numerical Features) ---")
print(df[numerical_for_plot].corr())
plt.figure(figsize=(7, 6))
sns.heatmap(df[numerical_for_plot].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()


# --- Step 3: Define Features (X) and Target (y) ---
X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN]

# Identify numerical and categorical features for preprocessing
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist() # 'object' typically means string/category

print(f"\nFeatures selected: {FEATURE_COLUMNS}")
print(f"Target selected: {TARGET_COLUMN}")
print(f"Numerical features for preprocessing: {numerical_features}")
print(f"Categorical features for preprocessing: {categorical_features}")

# --- Step 4: Data Preprocessing Pipeline ---
# This pipeline applies StandardScaler to numerical features and OneHotEncoder to categorical features.
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough' # Keep other columns not specified (if any)
)

# --- Step 5: Split Data into Training and Testing Sets ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# --- Step 6: Model Training and Evaluation ---

# Initialize different regression models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest Regressor': RandomForestRegressor(random_state=42),
    'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\n--- Training {name} ---")

    # Create a pipeline that first preprocesses, then trains the model
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', model)])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    results[name] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}

    print(f"{name} Evaluation:")
    print(f"  Mean Absolute Error (MAE): {mae:.2f}")
    print(f"  Mean Squared Error (MSE): {mse:.2f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"  R-squared (R2): {r2:.2f}")

# --- Step 7: Compare Model Results ---
print("\n--- Model Comparison ---")
# Sort models by R2 score (highest is best for R2)
sorted_results = sorted(results.items(), key=lambda item: item[1]['R2'], reverse=True)
for name, metrics in sorted_results:
    print(f"\n{name}:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.2f}")

# --- Step 8: Make a Prediction with the Best Model (Example) ---
best_model_name = sorted_results[0][0] # Get the name of the model with the highest R2
best_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', models[best_model_name])])
best_pipeline.fit(X_train, y_train) # Ensure the pipeline for the best model is fitted

print(f"\n--- Making a Prediction using the Best Model: {best_model_name} ---")

# Example new student data
# IMPORTANT: Ensure the column names and data types match your FEATURE_COLUMNS list.
# For 'gender', use the exact string 'Female' or 'Male' as in your training data.
new_student_data = pd.DataFrame({
    'hours_studied': [9],
    'previous_score': [70],
    'attendance_percentage': [88],
    'gender': ['Male']
})

predicted_score = best_pipeline.predict(new_student_data)
print(f"Predicted exam score for the new student (Hours: 9, Prev: 70, Att: 88%, Gender: Male): {predicted_score[0]:.2f}")

# Another example: A female student who studied a lot
new_student_data_2 = pd.DataFrame({
    'hours_studied': [16],
    'previous_score': [92],
    'attendance_percentage': [99],
    'gender': ['Female']
})
predicted_score_2 = best_pipeline.predict(new_student_data_2)
print(f"Predicted exam score for another student (Hours: 16, Prev: 92, Att: 99%, Gender: Female): {predicted_score_2[0]:.2f}")

print("\n--- Project Complete ---")
print("You can now experiment with different features, models, or new student data.")
print("Remember to install necessary libraries if you haven't already: pip install pandas scikit-learn matplotlib seaborn numpy")