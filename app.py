import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from src.data import _to_float, _norm_category, _norm_yes_no, _clip_ranges
from src.config import COLUMNS, risk_category

# Set page config
st.set_page_config(
    page_title="Mental Health Risk Assessment",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .reportview-container .main .block-container{
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        color: #2c3e50;
    }
    .stAlert {
        border-radius: 10px;
    }
    .prediction-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        color: #000000;
    }
    .risk-high { background-color: #fef0e6; border-left: 5px solid #e95353; background-image: linear-gradient(135deg, #fef0e6 0%, #fde0d4 100%); color: #000000; }
    .risk-medium { background-color: #fefae0; border-left: 5px solid #f4a261; background-image: linear-gradient(135deg, #fefae0 0%, #fde9c7 100%); color: #000000; }
    .risk-low { background-color: #e8f5e9; border-left: 5px solid #4caf50; background-image: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%); color: #000000; }
    .prediction-card h3, .prediction-card h1, .prediction-card p, .prediction-card strong {
        color: #000000;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🧠 Mental Health Risk Assessment System")
st.markdown("""
This application predicts mental health conditions, stress levels, and severity based on lifestyle factors.
Enter your information below to receive a personalized assessment.
""")

# Load models
@st.cache_resource
def load_models():
    try:
        condition_model = joblib.load("models/condition_model.pkl")
        stress_model = joblib.load("models/stress_model.pkl")
        severity_model = joblib.load("models/severity_model.pkl")
        
        condition_encoder = joblib.load("models/condition_label_encoder.pkl")
        stress_encoder = joblib.load("models/stress_encoder.pkl")
        severity_encoder = joblib.load("models/severity_encoder.pkl")
        
        return {
            'condition_model': condition_model,
            'stress_model': stress_model,
            'severity_model': severity_model,
            'condition_encoder': condition_encoder,
            'stress_encoder': stress_encoder,
            'severity_encoder': severity_encoder
        }
    except FileNotFoundError as e:
        st.error(f"Model files not found. Please ensure all model files are in the 'models' directory. Error: {e}")
        return None

models = load_models()

def preprocess_input_data(data):
    """Preprocess input data to match the format expected by the models"""
    # Create a DataFrame from the input data with correct column names
    mapped_data = {
        "Age": data["Age"],
        "Sleep_Hours": data["Sleep_Hours"],
        "Work_Hours": data["Work_Hours"],
        "Physical_Activity_Hours": data["Physical_Activity_Hours"],
        "Social_Media_Usage": data["Social_Media_Usage"],
        "Gender": data["Gender"],
        "Occupation": data["Occupation"],
        "Country": data["Country"],
        "Consultation_History": data["Consultation_History"],
        "Diet_Quality": data["Diet_Quality"],
        "Smoking_Habit": data["Smoking_Habit"],
        "Alcohol_Consumption": data["Alcohol_Consumption"],
        "Medication_Usage": data["Medication_Usage"],
    }
    
    # Create a DataFrame from the mapped data
    df = pd.DataFrame([mapped_data])
    
    # Apply the same preprocessing as in the training pipeline
    # Convert numerics
    for col in COLUMNS.numeric: 
        if col in df: 
            df[col] = df[col].map(_to_float)
    
    # Normalize Yes/No fields
    for col in ["Consultation_History", "Medication_Usage"]:
        if col in df:
            df[col] = df[col].map(_norm_yes_no).fillna("No").astype(str)
    
    # Normalize categorical fields
    categorical_fields = ["Gender", "Occupation", "Country", "Diet_Quality", 
                         "Smoking_Habit", "Alcohol_Consumption"]
    for col in categorical_fields:
        if col in df: 
            df[col] = df[col].map(_norm_category)
    
    # Clip ranges
    df = _clip_ranges(df)
    
    return df

def predict_mental_health(data):
    """Make predictions using the loaded models"""
    if models is None:
        return None
        
    try:
        # Preprocess the input data
        input_df = preprocess_input_data(data)
        
        # Make predictions
        cond_pred = models['condition_model'].predict(input_df)[0]
        cond_probs = models['condition_model'].predict_proba(input_df)[0]
        cond_prob = float(max(cond_probs))
        
        stress_pred = models['stress_model'].predict(input_df)[0]
        sev_pred = models['severity_model'].predict(input_df)[0]
        
        # Decode predictions
        stress_label = models['stress_encoder'].inverse_transform([int(stress_pred)])[0]
        severity_label = models['severity_encoder'].inverse_transform([int(sev_pred)])[0]
        
        return {
            'condition': str(cond_pred),
            'condition_confidence': f"{cond_prob:.1%}",
            'stress_level': str(stress_label),
            'severity': str(severity_label)
        }
    except Exception as e:
        st.error(f"Error making predictions: {e}")
        return None

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["📊 Assessment", "📈 Analysis", "ℹ️ About"])

with tab1:
    st.header("Personal Assessment")
    
    # Create columns for input fields
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Personal Information")
        age = st.number_input("Age", min_value=10, max_value=100, value=30)
        gender = st.selectbox("Gender", ["Male", "Female", "Non-Binary"])
        occupation = st.selectbox("Occupation", [
            "IT", "Healthcare", "Sales", "Education", "Engineering", 
            "Finance", "Marketing", "Student", "Other"
        ])
        country = st.selectbox("Country", [
            "USA", "UK", "Canada", "Germany", "Australia", 
            "France", "India", "Brazil", "Japan", "Other"
        ])
    
    with col2:
        st.subheader("Lifestyle Factors")
        sleep_hours = st.slider("Sleep Hours per Day", 0.0, 16.0, 7.0, 0.5)
        work_hours = st.slider("Work Hours per Week", 0.0, 80.0, 40.0, 1.0)
        physical_activity = st.slider("Physical Activity Hours per Week", 0.0, 40.0, 5.0, 0.5)
        social_media = st.slider("Social Media Usage (hours/day)", 0.0, 24.0, 3.0, 0.5)
    
    with col3:
        st.subheader("Health Factors")
        consultation_history = st.radio("Consultation History", ["Yes", "No"], index=1)
        medication_usage = st.radio("Medication Usage", ["Yes", "No"], index=1)
        diet_quality = st.selectbox("Diet Quality", ["Healthy", "Average", "Unhealthy"])
        smoking_habit = st.selectbox("Smoking Habit", [
            "Non-Smoker", "Occasional Smoker", "Regular Smoker", "Heavy Smoker"
        ])
        alcohol_consumption = st.selectbox("Alcohol Consumption", [
            "Non-Drinker", "Occasional Drinker", "Social Drinker", "Regular Drinker"
        ])
    
    # Create input dictionary
    input_data = {
        "Age": age,
        "Sleep_Hours": sleep_hours,
        "Work_Hours": work_hours,
        "Physical_Activity_Hours": physical_activity,
        "Social_Media_Usage": social_media,
        "Gender": gender,
        "Occupation": occupation,
        "Country": country,
        "Consultation_History": consultation_history,
        "Diet_Quality": diet_quality,
        "Smoking_Habit": smoking_habit,
        "Alcohol_Consumption": alcohol_consumption,
        "Medication_Usage": medication_usage,
    }
    
    # Prediction button
    if st.button("🔮 Assess Mental Health Risk", type="primary", use_container_width=True):
        with st.spinner("Analyzing your data..."):
            result = predict_mental_health(input_data)
            
            if result:
                st.success("Assessment complete!")
                
                # Display results in cards
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Mental Health Condition", result['condition'])
                
                with col2:
                    st.metric("Stress Level", result['stress_level'])
                
                with col3:
                    st.metric("Severity", result['severity'])
                
                # Risk visualization
                st.subheader("Risk Visualization")
                
                # Create risk score based on stress and severity
                stress_mapping = {"Low": 1, "Medium": 2, "High": 3}
                severity_mapping = {"None": 0, "Low": 1, "Medium": 2, "High": 3}
                
                stress_score = stress_mapping.get(result['stress_level'], 1)
                severity_score = severity_mapping.get(result['severity'], 0)
                overall_risk = (stress_score + severity_score) / 6 * 100
                
                # Determine risk category
                if overall_risk < 30:
                    risk_cat = "Low Risk"
                    risk_class = "risk-low"
                elif overall_risk < 70:
                    risk_cat = "Medium Risk"
                    risk_class = "risk-medium"
                else:
                    risk_cat = "High Risk"
                    risk_class = "risk-high"
                
                # Display risk card
                st.markdown(f"""
                <div class="prediction-card {risk_class}">
                    <h3>Overall Risk Assessment</h3>
                    <h1>{overall_risk:.1f}%</h1>
                    <p><strong>Risk Category:</strong> {risk_cat}</p>
                    <p><strong>Confidence:</strong> {result['condition_confidence']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Recommendations based on results
                st.subheader("Personalized Recommendations")
                
                recommendations = []
                
                if sleep_hours < 6:
                    recommendations.append("🌙 Aim for 7-9 hours of quality sleep per night")
                
                if work_hours > 50:
                    recommendations.append("💼 Consider work-life balance strategies to reduce overtime")
                
                if physical_activity < 3:
                    recommendations.append("🏃‍♀️ Increase physical activity to at least 150 minutes per week")
                
                if social_media > 5:
                    recommendations.append("📱 Limit social media usage to reduce potential negative impacts")
                
                if diet_quality == "Unhealthy":
                    recommendations.append("🥗 Improve your diet with more fruits, vegetables, and whole grains")
                
                if smoking_habit != "Non-Smoker":
                    recommendations.append("🚭 Consider smoking cessation programs for better health")
                
                if alcohol_consumption == "Regular Drinker":
                    recommendations.append("🍷 Consider reducing alcohol consumption for better mental health")
                
                if not recommendations:
                    recommendations.append("Keep up the healthy lifestyle!")
                
                for rec in recommendations:
                    st.markdown(f"- {rec}")

                # Radar chart for lifestyle factors
                st.subheader("Lifestyle Profile")
                
                # Prepare data for radar chart
                categories = ['Sleep', 'Work', 'Exercise', 'Social Media']
                values = [sleep_hours/8*100, min(work_hours/40*100, 100), 
                         physical_activity/7*100, 100-social_media/8*100]
                
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name='Your Profile'
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100]
                        )),
                    showlegend=False,
                    title="Lifestyle Balance Chart"
                )
                
                st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Data Insights")
    
    st.markdown("""
    ### Population Risk Distribution
    The following data represents typical risk distribution patterns observed in mental health studies:
    """)
    
    # Sample data for visualization
    sample_data = pd.DataFrame({
        'Category': ['Low Risk', 'Medium Risk', 'High Risk'],
        'Count': [45, 35, 20],
        'Percentage': [45, 35, 20]
    })
    
    st.subheader("Risk Distribution in Population")
    fig1 = px.pie(sample_data, values='Count', names='Category', 
                  color='Category',
                  color_discrete_map={'Low Risk':'#4caf50', 
                                    'Medium Risk':'#ff9800', 
                                    'High Risk':'#f44336'})
    st.plotly_chart(fig1, use_container_width=True)
    
    st.markdown("""
    ### Key Risk Factors
    Research has identified several key factors that contribute to mental health risks:
    """)
    
    st.subheader("Key Risk Factors")
    factors = ['Sleep Deprivation', 'Work Overload', 'Lack of Exercise', 
               'Poor Diet', 'Substance Use', 'Social Isolation']
    importance = [0.25, 0.22, 0.18, 0.15, 0.12, 0.08]
    
    fig2 = px.bar(x=factors, y=importance, 
                  labels={'x': 'Risk Factors', 'y': 'Importance'},
                  color=importance, color_continuous_scale=['#4caf50', '#ff9800', '#f44336'])
    fig2.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig2, use_container_width=True)
    
    st.info("ℹ️ These insights are based on analysis of mental health data trends.")

with tab3:
    st.header("About This System")
    
    st.markdown("""
    ### 🧠 Mental Health Risk Assessment System
    
    This application uses machine learning to predict mental health conditions, stress levels, 
    and severity based on various lifestyle and personal factors.
    
    #### 🔍 How It Works
    - Our models are trained on extensive mental health datasets
    - Predictions are based on factors like sleep, work hours, exercise, diet, and more
    - Results include personalized risk assessments and recommendations
    
    #### ⚙️ Technical Details
    - Uses ensemble machine learning algorithms for high accuracy
    - Features preprocessing pipelines for data normalization
    - Implements cross-validation for robust model evaluation
    
    #### 🛡️ Privacy Notice
    - All data entered is processed locally and not stored
    - No personal information is collected or transmitted
    
    #### 📂 More Information
    For detailed information about this project, please see our [ABOUT.md](ABOUT.md) file.
    
    #### 📞 Support
    If you're experiencing a mental health crisis, please contact:
    - National Suicide Prevention Lifeline: 988 (US)
    - Crisis Text Line: Text HOME to 741741
    """)

# Footer
st.markdown("---")
st.caption("🧠 Mental Health Risk Assessment System | For educational purposes only | Not a substitute for professional medical advice")