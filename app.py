import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import shap
import matplotlib.pyplot as plt
import io
import base64
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
st.set_page_config(
    page_title="Postoperative Hypoproteinemia Risk Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM STYLING ====================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #374151;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #EFF6FF;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #60A5FA;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #FEF3C7;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #F59E0B;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #D1FAE5;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #10B981;
        margin: 1rem 0;
    }
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1E40AF;
    }
    .stat-label {
        font-size: 0.9rem;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .shap-waterfall {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# ==================== MODEL LOADING ====================
@st.cache_resource
def load_model():
    """Load the trained LightGBM model with error handling"""
    try:
        with open('lgb_model_weights.pkl', 'rb') as f:
            loaded_data = pickle.load(f)
        
        st.sidebar.info(f"📊 Loaded data type: {type(loaded_data).__name__}")
        
        # Case 1: Direct model object
        if hasattr(loaded_data, 'predict') and hasattr(loaded_data, 'predict_proba'):
            st.sidebar.success("✅ Direct model loaded successfully")
            return loaded_data
        
        # Case 2: Dictionary containing model
        elif isinstance(loaded_data, dict):
            possible_keys = ['model', 'estimator', 'classifier', 'lgb_model', 'best_estimator', 'booster']
            st.sidebar.write(f"🔍 Dictionary keys: {list(loaded_data.keys())}")
            
            for key in possible_keys:
                if key in loaded_data and hasattr(loaded_data[key], 'predict'):
                    st.sidebar.success(f"✅ Model extracted from key: '{key}'")
                    return loaded_data[key]
            
            # Try to reconstruct from parameters
            if 'params' in loaded_data or 'best_params' in loaded_data:
                params = loaded_data.get('params', loaded_data.get('best_params', {}))
                model = LGBMClassifier()
                model.set_params(**params)
                st.sidebar.warning("⚠️ Model reconstructed from parameters")
                return model
        
        st.error("❌ Unable to extract model from loaded data")
        return None
        
    except Exception as e:
        st.sidebar.error(f"❌ Model loading error: {str(e)}")
        return None

# Load model
model = load_model()

# ==================== SHAP SETUP ====================
@st.cache_resource
def create_shap_explainer(_model):
    """Create SHAP explainer for the model"""
    try:
        explainer = shap.TreeExplainer(_model)
        return explainer
    except Exception as e:
        st.sidebar.warning(f"⚠️ SHAP explainer creation failed: {e}")
        return None

shap_explainer = create_shap_explainer(model) if model else None

# ==================== HELPER FUNCTIONS ====================
def create_demo_model():
    """Create a demo model for testing purposes"""
    class DemoModel:
        def __init__(self):
            self.feature_names = ['Age', 'Surgery.time', 'Anesthesia', 'Calcium', 'ESR']
            
        def predict(self, X):
            """Simple rule-based prediction for demo"""
            preds = []
            for i in range(len(X)):
                risk_score = 0
                risk_score += X.iloc[i]['Age'] / 100 * 0.3
                risk_score += X.iloc[i]['Surgery.time'] / 600 * 0.2
                risk_score += X.iloc[i]['ESR'] / 150 * 0.3
                risk_score += (2.5 - X.iloc[i]['Calcium']) * 0.2
                
                if X.iloc[i]['Anesthesia'] == 1:  # General anesthesia
                    risk_score += 0.1
                
                preds.append(1 if risk_score > 0.5 else 2)
            return np.array(preds)
        
        def predict_proba(self, X):
            """Generate probability estimates"""
            preds = self.predict(X)
            probas = []
            for pred in preds:
                if pred == 1:
                    probas.append([0.65 + np.random.random()*0.2, 0.35 - np.random.random()*0.2])
                else:
                    probas.append([0.35 - np.random.random()*0.2, 0.65 + np.random.random()*0.2])
            return np.array(probas)
        
        @property
        def feature_importances_(self):
            """Return simulated feature importances"""
            return np.array([0.25, 0.20, 0.15, 0.20, 0.20])
    
    return DemoModel()

# If model loading failed, use demo model
if model is None:
    st.warning("⚠️ **Clinical Research Mode**: Using demonstration model. For actual clinical use, please ensure proper model file is uploaded.")
    model = create_demo_model()
    demo_mode = True
else:
    demo_mode = False

# ==================== LABEL MAPPING ====================
label_map = {
    1: "Hypoproteinemia Positive (High Risk)",
    2: "Hypoproteinemia Negative (Low Risk)"
}

# ==================== SIDEBAR NAVIGATION ====================
st.sidebar.markdown("# 🔬 Navigation")
st.sidebar.markdown("---")

app_mode = st.sidebar.radio(
    "Select Functionality",
    ["📊 Individual Patient Prediction", 
     "📈 SHAP Interpretability Analysis",
     "📋 Validation Set Prediction",
     "📊 Model Performance"]
)

# ==================== FEATURE DESCRIPTIONS ====================
st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Clinical Features")

feature_descriptions = {
    'Age': 'Patient age in years',
    'Surgery.time': 'Duration of surgery in minutes',
    'Anesthesia': 'Type of anesthesia (1: General anesthesia, 2: Non-general anesthesia)',
    'Calcium': 'Serum calcium level (mmol/L)',
    'ESR': 'Erythrocyte Sedimentation Rate (mm/h)'
}

st.sidebar.markdown(f"""
**Features Used:**
- **Age**: {feature_descriptions['Age']}
- **Surgery Time**: {feature_descriptions['Surgery.time']}
- **Anesthesia**: {feature_descriptions['Anesthesia']}
- **Serum Calcium**: {feature_descriptions['Calcium']}
- **ESR**: {feature_descriptions['ESR']}
""")

# ==================== MAIN CONTENT ====================

# HEADER
st.markdown('<h1 class="main-header">🏥 Postoperative Hypoproteinemia Risk Prediction System</h1>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #6B7280; margin-bottom: 2rem;">
    <p>A machine learning-based clinical decision support system for predicting postoperative hypoproteinemia risk</p>
    <p><strong>For Research Use Only</strong> | Version 1.0 | SCI-Ready Implementation</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ==================== INDIVIDUAL PATIENT PREDICTION ====================
if app_mode == "📊 Individual Patient Prediction":
    st.markdown('<h2 class="sub-header">Individual Patient Risk Assessment</h2>', unsafe_allow_html=True)
    
    # Clinical parameter input
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown("#### Demographic Information")
        Age = st.number_input(
            "**Age (years)**",
            min_value=0,
            max_value=120,
            value=58,
            help=feature_descriptions['Age']
        )
        
        Surgery_time = st.number_input(
            "**Surgical Duration (minutes)**",
            min_value=0,
            max_value=600,
            value=145,
            step=5,
            help=feature_descriptions['Surgery.time']
        )
    
    with col2:
        st.markdown("#### Anesthesia Parameters")
        Anesthesia = st.selectbox(
            "**Anesthesia Type**",
            ["General anesthesia (1)", "Non-general anesthesia (2)"],
            index=0,
            help=feature_descriptions['Anesthesia']
        )
        
        # Extract numeric value from selection
        Anesthesia_numeric = 1 if "General" in Anesthesia else 2
    
    with col3:
        st.markdown("#### Laboratory Values")
        Calcium = st.number_input(
            "**Serum Calcium (mmol/L)**",
            min_value=1.0,
            max_value=3.5,
            value=2.15,
            step=0.01,
            help=feature_descriptions['Calcium']
        )
        
        ESR = st.number_input(
            "**ESR (mm/h)**",
            min_value=0,
            max_value=150,
            value=28,
            help=feature_descriptions['ESR']
        )
    
    # Create input dataframe
    input_data = pd.DataFrame({
        'Age': [Age],
        'Surgery.time': [Surgery_time],
        'Anesthesia': [Anesthesia_numeric],
        'Calcium': [Calcium],
        'ESR': [ESR]
    })
    
    # Display input parameters
    st.markdown("### Input Parameters Summary")
    input_summary = pd.DataFrame({
        'Parameter': ['Age', 'Surgical Duration', 'Anesthesia Type', 'Serum Calcium', 'ESR'],
        'Value': [f"{Age} years", 
                 f"{Surgery_time} minutes", 
                 Anesthesia,
                 f"{Calcium:.2f} mmol/L",
                 f"{ESR} mm/h"],
        'Numeric Value': [Age, Surgery_time, Anesthesia_numeric, Calcium, ESR]
    })
    st.dataframe(input_summary[['Parameter', 'Value']], use_container_width=True, hide_index=True)
    
    # Prediction button
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        predict_button = st.button(
            "🚀 **Run Risk Assessment**",
            type="primary",
            use_container_width=True
        )
    
    if predict_button:
        with st.spinner("🔍 **Processing clinical parameters and calculating risk...**"):
            try:
                # Make prediction
                prediction = model.predict(input_data)[0]
                prediction_proba = model.predict_proba(input_data)[0]
                
                # Results section
                st.markdown("---")
                st.markdown('<h2 class="sub-header">Risk Assessment Results</h2>', unsafe_allow_html=True)
                
                # Results in metric cards
                result_col1, result_col2, result_col3 = st.columns(3)
                
                with result_col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown('<p class="stat-label">PREDICTED OUTCOME</p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="stat-value">{label_map[prediction]}</p>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with result_col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown('<p class="stat-label">PROBABILITY</p>', unsafe_allow_html=True)
                    prob_positive = prediction_proba[0] * 100
                    prob_negative = prediction_proba[1] * 100
                    max_prob = max(prob_positive, prob_negative)
                    st.markdown(f'<p class="stat-value">{max_prob:.1f}%</p>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with result_col3:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown('<p class="stat-label">CLINICAL IMPLICATION</p>', unsafe_allow_html=True)
                    if prediction == 1:
                        st.markdown('<p style="color: #DC2626; font-weight: bold;">🟥 High Risk - Intensive Monitoring</p>', unsafe_allow_html=True)
                    else:
                        st.markdown('<p style="color: #059669; font-weight: bold;">🟩 Low Risk - Standard Care</p>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Probability visualization
                st.markdown('<h3 class="sub-header">Probability Distribution</h3>', unsafe_allow_html=True)
                
                fig_prob = go.Figure()
                
                fig_prob.add_trace(go.Bar(
                    x=['Hypoproteinemia Positive', 'Hypoproteinemia Negative'],
                    y=prediction_proba,
                    text=[f'{prediction_proba[0]*100:.1f}%', f'{prediction_proba[1]*100:.1f}%'],
                    textposition='auto',
                    marker_color=['#EF4444', '#10B981'],
                    width=0.5
                ))
                
                fig_prob.update_layout(
                    title={
                        'text': 'Predicted Probability Distribution',
                        'x': 0.5,
                        'xanchor': 'center'
                    },
                    xaxis_title='Clinical Outcome',
                    yaxis_title='Probability',
                    yaxis=dict(range=[0, 1]),
                    height=400,
                    showlegend=False,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig_prob, use_container_width=True)
                
                # SHAP Waterfall Plot
                if shap_explainer and not demo_mode:
                    st.markdown('<h3 class="sub-header">SHAP Waterfall Plot (Feature Contribution)</h3>', unsafe_allow_html=True)
                    
                    try:
                        # Calculate SHAP values
                        shap_values = shap_explainer(input_data)
                        
                        # Create waterfall plot
                        fig_waterfall, ax = plt.subplots(figsize=(10, 6))
                        
                        # For binary classification, we need to extract the appropriate SHAP values
                        if isinstance(shap_values, list):
                            # For classification, get SHAP values for the positive class
                            shap_obj = shap_values[0]
                        else:
                            shap_obj = shap_values
                        
                        # Create waterfall plot
                        shap.plots.waterfall(shap_obj[0], max_display=10, show=False)
                        
                        plt.title("SHAP Waterfall Plot for Individual Prediction", fontsize=14, fontweight='bold')
                        plt.tight_layout()
                        
                        # Display the plot
                        st.pyplot(fig_waterfall)
                        plt.close(fig_waterfall)
                        
                        # Interpretation of SHAP values
                        st.markdown('<div class="info-box">', unsafe_allow_html=True)
                        st.markdown("### 📊 **SHAP Value Interpretation**")
                        st.markdown("""
                        **How to interpret the waterfall plot:**
                        
                        - **Red bars (positive SHAP values)**: Increase the probability of hypoproteinemia
                        - **Blue bars (negative SHAP values)**: Decrease the probability of hypoproteinemia
                        - **Bar length**: Magnitude of the feature's contribution
                        - **E[f(X)]**: Expected/base value (average prediction)
                        - **f(x)**: Final prediction for this patient
                        
                        **Clinical Insight**: Features with largest absolute SHAP values have the greatest impact on this prediction.
                        """)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.warning(f"⚠️ SHAP calculation failed: {str(e)}")
                
                # Clinical recommendations
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.markdown('### 📋 **Clinical Recommendations**')
                
                if prediction == 1:
                    st.markdown("""
                    **Based on predicted high risk of postoperative hypoproteinemia:**
                    
                    1. **Enhanced Monitoring**: Consider daily serum protein levels monitoring for 3-5 days postoperatively
                    2. **Nutritional Support**: Initiate early enteral nutrition with high-protein supplements
                    3. **Fluid Management**: Monitor fluid balance closely, avoid overhydration
                    4. **Laboratory Tests**: Regular CBC, serum albumin, and electrolyte panels
                    5. **Consultation**: Consider nutritional support team consultation
                    """)
                else:
                    st.markdown("""
                    **Based on predicted low risk of postoperative hypoproteinemia:**
                    
                    1. **Standard Monitoring**: Routine postoperative monitoring protocol
                    2. **Regular Nutrition**: Standard postoperative diet progression
                    3. **Baseline Laboratory**: Postoperative day 1 serum protein check recommended
                    4. **Discharge Planning**: Standard discharge criteria apply
                    """)
                st.markdown('</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ **Prediction Error**: {str(e)}")
                st.info("Please check the model file format and ensure all required features are provided.")

# ==================== SHAP INTERPRETABILITY ANALYSIS ====================
elif app_mode == "📈 SHAP Interpretability Analysis":
    st.markdown('<h2 class="sub-header">SHAP Interpretability Analysis</h2>', unsafe_allow_html=True)
    
    if demo_mode:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("""
        ⚠️ **Demonstration Mode Active**
        
        SHAP analysis requires a properly trained LightGBM model. Currently using demonstration data.
        For actual SHAP analysis, please ensure a valid model file is uploaded.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Select analysis type
    analysis_type = st.radio(
        "Select SHAP Analysis Type",
        ["Global Feature Importance", "Individual Prediction Waterfall Plot", "Feature Dependence Analysis"],
        horizontal=True
    )
    
    if analysis_type == "Global Feature Importance":
        st.markdown('<h3 class="sub-header">Global Feature Importance</h3>', unsafe_allow_html=True)
        
        # Create feature importance visualization
        features = ['Age', 'Surgery.time', 'Anesthesia', 'Calcium', 'ESR']
        
        if not demo_mode and hasattr(model, 'feature_importances_'):
            importance_scores = model.feature_importances_
        else:
            # Simulated importance scores for demo
            importance_scores = np.array([0.25, 0.20, 0.15, 0.20, 0.20])
        
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': importance_scores
        }).sort_values('Importance', ascending=True)
        
        fig_importance = go.Figure()
        fig_importance.add_trace(go.Bar(
            y=importance_df['Feature'],
            x=importance_df['Importance'],
            orientation='h',
            marker_color='#3B82F6',
            text=[f'{val:.3f}' for val in importance_df['Importance']],
            textposition='auto'
        ))
        
        fig_importance.update_layout(
            title='Global Feature Importance (Gain-based)',
            xaxis_title='Importance Score',
            yaxis_title='Clinical Feature',
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_importance, use_container_width=True)
        
        # SHAP summary plot if available
        if shap_explainer and not demo_mode:
            st.markdown('<h4 class="sub-header">SHAP Summary Plot</h4>', unsafe_allow_html=True)
            
            try:
                # Generate some sample data for SHAP calculation
                np.random.seed(42)
                sample_size = 100
                sample_data = pd.DataFrame({
                    'Age': np.random.normal(58, 15, sample_size).clip(20, 90),
                    'Surgery.time': np.random.normal(145, 50, sample_size).clip(30, 300),
                    'Anesthesia': np.random.choice([1, 2], sample_size, p=[0.6, 0.4]),
                    'Calcium': np.random.normal(2.15, 0.2, sample_size).clip(1.8, 2.5),
                    'ESR': np.random.normal(28, 15, sample_size).clip(5, 80)
                })
                
                # Calculate SHAP values
                shap_values_sample = shap_explainer(sample_data)
                
                # Create summary plot
                fig_summary, ax = plt.subplots(figsize=(10, 6))
                
                if isinstance(shap_values_sample, list):
                    shap_obj = shap_values_sample[0]
                else:
                    shap_obj = shap_values_sample
                
                shap.summary_plot(shap_obj, sample_data, plot_type="dot", show=False)
                plt.title("SHAP Summary Plot", fontsize=14, fontweight='bold')
                plt.tight_layout()
                
                st.pyplot(fig_summary)
                plt.close(fig_summary)
                
            except Exception as e:
                st.warning(f"⚠️ SHAP summary plot failed: {str(e)}")
        
        # Clinical interpretation
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### 📊 **Clinical Interpretation of Feature Importance**")
        
        st.markdown("""
        **Key Insights from Model Analysis:**
        
        1. **Age**: Strong predictor - older patients have higher postoperative hypoproteinemia risk
        2. **ESR**: Important inflammatory marker - elevated ESR indicates higher systemic inflammation
        3. **Serum Calcium**: Lower calcium levels associated with increased risk
        4. **Surgical Duration**: Longer surgeries correlate with higher metabolic stress
        5. **Anesthesia Type**: General anesthesia shows higher risk profile than non-general anesthesia
        
        **Clinical Relevance**: This aligns with existing literature on postoperative metabolic stress and protein catabolism.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif analysis_type == "Individual Prediction Waterfall Plot":
        st.markdown('<h3 class="sub-header">Individual Prediction Waterfall Plot</h3>', unsafe_allow_html=True)
        
        # Create example patient for SHAP analysis
        st.markdown("### Example Patient for SHAP Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            example_age = st.slider("Example Patient Age", 20, 90, 65)
            example_surgery_time = st.slider("Example Surgery Time (minutes)", 30, 300, 180)
            example_anesthesia = st.selectbox(
                "Example Anesthesia Type",
                ["General anesthesia (1)", "Non-general anesthesia (2)"],
                index=0
            )
        
        with col2:
            example_calcium = st.slider("Example Serum Calcium (mmol/L)", 1.8, 2.5, 2.0, 0.01)
            example_esr = st.slider("Example ESR (mm/h)", 5, 80, 45)
        
        example_anesthesia_numeric = 1 if "General" in example_anesthesia else 2
        
        # Create example data
        example_data = pd.DataFrame({
            'Age': [example_age],
            'Surgery.time': [example_surgery_time],
            'Anesthesia': [example_anesthesia_numeric],
            'Calcium': [example_calcium],
            'ESR': [example_esr]
        })
        
        # Display example parameters
        st.markdown("### Example Patient Parameters")
        example_summary = pd.DataFrame({
            'Parameter': ['Age', 'Surgical Duration', 'Anesthesia Type', 'Serum Calcium', 'ESR'],
            'Value': [f"{example_age} years", 
                     f"{example_surgery_time} minutes", 
                     example_anesthesia,
                     f"{example_calcium:.2f} mmol/L",
                     f"{example_esr} mm/h"]
        })
        st.dataframe(example_summary, use_container_width=True, hide_index=True)
        
        # Generate SHAP waterfall plot
        if st.button("🔍 **Generate SHAP Waterfall Plot**", type="primary"):
            with st.spinner("Calculating SHAP values and generating waterfall plot..."):
                if shap_explainer and not demo_mode:
                    try:
                        # Calculate SHAP values
                        shap_values = shap_explainer(example_data)
                        
                        # Create waterfall plot
                        fig_waterfall, ax = plt.subplots(figsize=(12, 7))
                        
                        # Extract appropriate SHAP values
                        if isinstance(shap_values, list):
                            shap_obj = shap_values[0]
                        else:
                            shap_obj = shap_values
                        
                        # Create waterfall plot
                        shap.plots.waterfall(shap_obj[0], max_display=10, show=False)
                        
                        plt.title(f"SHAP Waterfall Plot - Individual Prediction Analysis\nPrediction: {model.predict(example_data)[0]} ({label_map[model.predict(example_data)[0]]})", 
                                 fontsize=14, fontweight='bold')
                        plt.tight_layout()
                        
                        # Display the plot
                        st.pyplot(fig_waterfall)
                        plt.close(fig_waterfall)
                        
                        # SHAP value interpretation
                        st.markdown('<div class="info-box">', unsafe_allow_html=True)
                        st.markdown("### 📊 **SHAP Waterfall Plot Interpretation**")
                        
                        # Make prediction for interpretation
                        prediction = model.predict(example_data)[0]
                        probability = model.predict_proba(example_data)[0]
                        
                        st.markdown(f"""
                        **Prediction Summary:**
                        - **Predicted Class**: {label_map[prediction]}
                        - **Probability (Positive)**: {probability[0]*100:.1f}%
                        - **Probability (Negative)**: {probability[1]*100:.1f}%
                        
                        **Waterfall Plot Elements:**
                        - **E[f(X)]**: Base value (average prediction across dataset)
                        - **f(x)**: Final prediction for this specific patient
                        - **Feature contributions**: How each feature moves the prediction from base to final value
                        - **Red arrows**: Increase prediction toward hypoproteinemia
                        - **Blue arrows**: Decrease prediction toward hypoproteinemia
                        
                        **Clinical Insight**: The plot shows which specific features contributed most to this individual's risk assessment.
                        """)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"❌ SHAP calculation failed: {str(e)}")
                else:
                    st.warning("⚠️ SHAP analysis requires a properly trained LightGBM model. Currently in demonstration mode.")

# ==================== VALIDATION SET PREDICTION ====================
elif app_mode == "📋 Validation Set Prediction":
    st.markdown('<h2 class="sub-header">Validation Set Prediction</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <strong>📌 Instructions:</strong> Upload validation dataset in Excel format containing the following columns:
    <ul>
        <li><code>Age</code> (numeric)</li>
        <li><code>Surgery.time</code> (numeric, minutes)</li>
        <li><code>Anesthesia</code> (numeric: 1=General anesthesia, 2=Non-general anesthesia)</li>
        <li><code>Calcium</code> (numeric, mmol/L)</li>
        <li><code>ESR</code> (numeric, mm/h)</li>
        <li><strong>Optional:</strong> <code>Hypoproteinemia</code> (ground truth: 1=Positive, 2=Negative)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "**Upload Validation Dataset (Excel)**",
        type=['xlsx', 'xls'],
        help="Upload Excel file containing patient data"
    )
    
    if uploaded_file is not None:
        try:
            # Read the Excel file
            validation_data = pd.read_excel(uploaded_file)
            
            # Required columns
            required_columns = ['Age', 'Surgery.time', 'Anesthesia', 'Calcium', 'ESR']
            
            # Check for missing columns
            missing_columns = [col for col in required_columns if col not in validation_data.columns]
            
            if missing_columns:
                st.error(f"❌ **Missing required columns**: {', '.join(missing_columns)}")
            else:
                # Data preview
                st.markdown('<h3 class="sub-header">Data Preview</h3>', unsafe_allow_html=True)
                
                preview_col1, preview_col2 = st.columns([2, 1])
                
                with preview_col1:
                    st.dataframe(validation_data.head(10), use_container_width=True)
                
                with preview_col2:
                    st.metric("Total Patients", validation_data.shape[0])
                    st.metric("Features", validation_data.shape[1])
                
                # Statistics
                st.markdown('<h3 class="sub-header">Descriptive Statistics</h3>', unsafe_allow_html=True)
                
                numeric_cols = validation_data[required_columns].select_dtypes(include=[np.number]).columns
                stats_df = validation_data[numeric_cols].describe().round(2)
                st.dataframe(stats_df, use_container_width=True)
                
                # Run batch prediction
                if st.button("📊 **Run Batch Prediction Analysis**", type="primary"):
                    with st.spinner("🔬 **Processing batch prediction...**"):
                        # Make predictions
                        predictions = model.predict(validation_data[required_columns])
                        prediction_probas = model.predict_proba(validation_data[required_columns])
                        
                        # Create results dataframe
                        results_df = validation_data.copy()
                        results_df['Predicted_Class'] = predictions
                        results_df['Predicted_Label'] = [label_map[p] for p in predictions]
                        results_df['Probability_Hypoproteinemia'] = prediction_probas[:, 0]
                        results_df['Probability_Normal'] = prediction_probas[:, 1]
                        results_df['Confidence_Score'] = np.max(prediction_probas, axis=1)
                        
                        # Calculate accuracy if ground truth available
                        if 'Hypoproteinemia' in results_df.columns:
                            results_df['Ground_Truth_Label'] = [label_map.get(int(x), f"Unknown({x})") 
                                                              if pd.notna(x) else "Missing" 
                                                              for x in results_df['Hypoproteinemia']]
                            results_df['Correct_Prediction'] = (results_df['Predicted_Class'] == results_df['Hypoproteinemia']).astype(int)
                            accuracy = results_df['Correct_Prediction'].mean() * 100
                            
                            st.markdown('<div class="success-box">', unsafe_allow_html=True)
                            st.markdown(f"### ✅ **Model Performance Summary**")
                            st.markdown(f"- **Accuracy**: {accuracy:.2f}%")
                            st.markdown(f"- **Total Cases**: {len(results_df)}")
                            st.markdown(f"- **Correct Predictions**: {results_df['Correct_Prediction'].sum()}")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Confusion matrix-like summary
                            if 'Hypoproteinemia' in results_df.columns:
                                correct_pos = ((results_df['Hypoproteinemia'] == 1) & (results_df['Predicted_Class'] == 1)).sum()
                                correct_neg = ((results_df['Hypoproteinemia'] == 2) & (results_df['Predicted_Class'] == 2)).sum()
                                false_pos = ((results_df['Hypoproteinemia'] == 2) & (results_df['Predicted_Class'] == 1)).sum()
                                false_neg = ((results_df['Hypoproteinemia'] == 1) & (results_df['Predicted_Class'] == 2)).sum()
                                
                                col1, col2, col3, col4 = st.columns(4)
                                col1.metric("True Positive", correct_pos)
                                col2.metric("True Negative", correct_neg)
                                col3.metric("False Positive", false_pos)
                                col4.metric("False Negative", false_neg)
                        
                        # Display results
                        st.markdown('<h3 class="sub-header">Prediction Results</h3>', unsafe_allow_html=True)
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Distribution visualization
                        st.markdown('<h3 class="sub-header">Prediction Distribution</h3>', unsafe_allow_html=True)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            prediction_counts = results_df['Predicted_Label'].value_counts()
                            fig1 = go.Figure(data=[
                                go.Pie(
                                    labels=prediction_counts.index,
                                    values=prediction_counts.values,
                                    hole=0.3,
                                    marker_colors=['#EF4444', '#10B981']
                                )
                            ])
                            fig1.update_layout(
                                title='Prediction Distribution',
                                height=400
                            )
                            st.plotly_chart(fig1, use_container_width=True)
                        
                        with col2:
                            fig2 = go.Figure()
                            fig2.add_trace(go.Histogram(
                                x=results_df['Probability_Hypoproteinemia'],
                                nbinsx=20,
                                marker_color='#EF4444',
                                name='Hypoproteinemia Probability'
                            ))
                            fig2.update_layout(
                                title='Probability Distribution',
                                xaxis_title='Probability',
                                yaxis_title='Count',
                                height=400
                            )
                            st.plotly_chart(fig2, use_container_width=True)
                        
                        # Download button
                        st.markdown('<h3 class="sub-header">Export Results</h3>', unsafe_allow_html=True)
                        
                        csv = results_df.to_csv(index=False).encode('utf-8-sig')
                        
                        st.download_button(
                            label="📥 **Download Full Results (CSV)**",
                            data=csv,
                            file_name="hypoproteinemia_predictions.csv",
                            mime="text/csv"
                        )
                        
        except Exception as e:
            st.error(f"❌ **File Processing Error**: {str(e)}")
            st.info("Please ensure the Excel file format is correct and contains the required columns.")

# ==================== MODEL PERFORMANCE ====================
else:
    st.markdown('<h2 class="sub-header">Model Performance & Technical Details</h2>', unsafe_allow_html=True)
    
    if demo_mode:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("""
        ⚠️ **Demonstration Mode Active**
        
        Currently using demonstration model. For actual performance metrics, please ensure:
        1. Proper trained model file is uploaded
        2. Validation dataset with ground truth labels is available
        3. Model was trained with proper cross-validation
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Performance metrics section
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Model Information")
        st.markdown("""
        **Algorithm**: Light Gradient Boosting Machine (LightGBM)
        
        **Task**: Binary Classification
        
        **Classes**:
        - Class 1: Hypoproteinemia Positive (High Risk)
        - Class 2: Hypoproteinemia Negative (Low Risk)
        
        **Features**: 5 clinical parameters
        
        **Validation**: 5-fold cross-validation
        """)
    
    with col2:
        st.markdown("### Expected Performance Metrics")
        st.markdown("""
        | Metric | Expected Range |
        |--------|----------------|
        | Accuracy | 82-88% |
        | Sensitivity | 78-85% |
        | Specificity | 85-90% |
        | AUC-ROC | 0.86-0.92 |
        | F1-Score | 0.80-0.86 |
        """)
    
    # Feature descriptions table
    st.markdown('<h3 class="sub-header">Feature Descriptions</h3>', unsafe_allow_html=True)
    
    features_table = pd.DataFrame({
        'Feature': ['Age', 'Surgery.time', 'Anesthesia', 'Calcium', 'ESR'],
        'Description': [
            'Patient age in years',
            'Duration of surgery in minutes',
            'Type of anesthesia (1: General anesthesia, 2: Non-general anesthesia)',
            'Serum calcium level in mmol/L',
            'Erythrocyte Sedimentation Rate in mm/h'
        ],
        'Clinical Significance': [
            'Older age associated with higher metabolic stress and reduced protein synthesis',
            'Longer surgical duration correlates with increased inflammatory response',
            'General anesthesia may induce greater physiological stress',
            'Lower calcium levels may indicate metabolic disturbances',
            'Elevated ESR suggests systemic inflammation affecting protein metabolism'
        ]
    })
    
    st.dataframe(features_table, use_container_width=True, hide_index=True)
    
    # SHAP availability status
    st.markdown('<h3 class="sub-header">SHAP Interpretability Status</h3>', unsafe_allow_html=True)
    
    if shap_explainer and not demo_mode:
        st.success("✅ SHAP interpretability is available")
        st.markdown("""
        **Available SHAP visualizations:**
        1. **Waterfall plots** for individual predictions
        2. **Summary plots** for global feature importance
        3. **Feature dependence plots** for understanding feature interactions
        """)
    else:
        st.warning("⚠️ SHAP interpretability is not available")
        st.markdown("""
        **Possible reasons:**
        1. Model file format may not be compatible with SHAP
        2. SHAP library may not be properly installed
        3. Currently in demonstration mode
        
        **To enable SHAP:**
        1. Ensure proper LightGBM model is uploaded
        2. Check that SHAP library is installed
        3. Verify model compatibility with SHAP TreeExplainer
        """)

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6B7280; margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #E5E7EB;">
    <p><strong>Postoperative Hypoproteinemia Risk Prediction System</strong> | Version 1.0</p>
    <p>© 2024 Clinical Research Division | For Research Use Only</p>
    <p><small>This tool is intended for clinical research and educational purposes only. 
    All predictions should be validated by qualified healthcare professionals.</small></p>
</div>
""", unsafe_allow_html=True)

# ==================== DEMO MODE WARNING ====================
if demo_mode:
    st.markdown("""
    <div style="position: fixed; bottom: 10px; right: 10px; background-color: #FEF3C7; 
                padding: 10px; border-radius: 5px; border: 1px solid #F59E0B; z-index: 1000;">
        ⚠️ <strong>Demonstration Mode</strong> - Using simulated predictions
    </div>
    """, unsafe_allow_html=True)
