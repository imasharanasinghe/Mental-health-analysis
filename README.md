# 🧠 Mental Health Risk Prediction & Assessment System

An advanced machine learning system for predicting mental health conditions, stress levels, and severity, featuring automated preprocessing, multi-algorithm evaluation, and an interactive web interface.

## 🌟 Features

- **Multi-Target Prediction**: Predicts mental health conditions, stress levels, and severity simultaneously
- **Risk Scoring**: Provides comprehensive risk assessment with confidence scores
- **Interactive Dashboard**: Streamlit-based web interface for easy interaction
- **Personalized Recommendations**: Actionable insights based on individual risk factors
- **Lifestyle Analysis**: Radar chart visualization of lifestyle balance
- **Data Insights**: Population-level risk distribution and key risk factors
- **Model Explainability**: SHAP/LIME integration for understanding predictions
- **Lifestyle "What-If" Scenarios**: Explore how lifestyle changes might affect risk
- **Anomaly Detection**: Identifies unusual patterns in mental health data
- **PDF Reports**: Generate detailed assessment reports
- **Model Comparison**: Cross-validation and hyperparameter tuning

## 🚀 Quick Start

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd mental-health-ml
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```

3. Activate the virtual environment:
   ```bash
   # Windows
   .venv\Scripts\activate
   
   # Mac/Linux
   source .venv/bin/activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

6. Open your browser to `http://localhost:8501` to use the application.

## 📁 Project Structure

```
mental-health-ml/
├── app.py                 # Main Streamlit application
├── src/                   # Source code for ML models and utilities
│   ├── data.py           # Data preprocessing functions
│   ├── modeling.py       # Model training and evaluation
│   ├── train_*.py        # Specific model training scripts
│   └── ...
├── models/               # Trained model files
├── data/                 # Dataset files
├── reports/              # Generated reports
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🧪 Testing

Run the test scripts to verify model functionality:

```bash
python test_predictions.py
```

## 📊 Data Requirements

The system expects a CSV file with the following columns:
- User_ID
- Age
- Sleep_Hours
- Work_Hours
- Physical_Activity_Hours
- Social_Media_Usage
- Gender
- Occupation
- Country
- Consultation_History
- Diet_Quality
- Smoking_Habit
- Alcohol_Consumption
- Medication_Usage
- Mental_Health_Condition (target)
- Stress_Level (target)
- Severity (target)

## 🤖 Models

The system uses ensemble machine learning algorithms:
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- Gradient Boosting
- XGBoost

Models are automatically evaluated using 5-fold cross-validation, and the best performing model is selected for each target variable.

## 🛡️ Privacy

All data processing is done locally. No personal information is collected or transmitted.

## 📞 Support

If you're experiencing a mental health crisis, please contact:
- National Suicide Prevention Lifeline: 988 (US)
- Crisis Text Line: Text HOME to 741741

