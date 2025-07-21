# Student Performance Predictor

The Student Performance Predictor is a machine learning project designed to forecast student exam scores based on key academic and behavioral indicators, such as study hours, previous grades, and attendance.  
This tool aims to support educators and institutions in making data-driven decisions to enhance student outcomes.

---

## Project Goals

- Identify which factors most significantly influence student performance.
- Proactively detect students at risk of underperforming.
- Inform decisions regarding resource allocation and personalized learning strategies.

---

## Key Features

- **Data Loading**: Seamlessly load datasets from CSV files for easy integration.
- **Exploratory Data Analysis (EDA)**:
  - Generate descriptive statistics.
  - Identify missing values.
  - Visualize distributions and relationships (histograms, scatter plots, correlation heatmaps).
- **Data Preprocessing**:
  - Scale numerical features using `StandardScaler`.
  - Encode categorical variables using `OneHotEncoder`.
- **Model Training and Comparison**:
  - Linear Regression
  - Random Forest Regressor
  - Gradient Boosting Regressor
- **Model Evaluation** using standard metrics:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - R-squared (R²)
- **Intelligent Model Selection**: Automatically identifies the best-performing model based on R² score.
- **Prediction Capability**: Use the trained model to predict exam scores for new student data.

---


## Project Structure 
student-performance-predictor/
├── data/                    # Dataset files (CSV, etc.)
├── src/                     # Source code modules
│   ├── data_loader.py       # Functions to load data
│   ├── eda.py               # Functions for exploratory data analysis
│   ├── preprocessing.py     # Data preprocessing utilities
│   ├── model_training.py    # Model training & evaluation
│   └── prediction.py       # Functions for making predictions
├── main.py                  # Entry point script to run the whole pipeline
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
└── .vscode/                 # VS Code workspace settings (optional)
    ├── settings.json
    └── launch.json
