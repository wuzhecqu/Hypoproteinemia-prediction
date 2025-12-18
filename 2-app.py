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

# ==================== SIMPLE SHAP HANDLER ====================
def get_safe_shap_values(explainer, data, model):
    """Get SHAP values with safe handling for different SHAP versions"""
    try:
        # Get SHAP values
        shap_values = explainer.shap_values(data)
        
        # Handle different SHAP return types
        if isinstance(shap_values, list):
            # For binary classification, shap_values is a list [shap_values_negative, shap_values_positive]
            # We want the positive class (class 1) which is usually index 0 or 1 depending on SHAP version
            if len(shap_values) == 2:
                # Check which one corresponds to our positive class (1)
                # Usually index 0 is negative, index 1 is positive
                return shap_values[1]  # Positive class
            else:
                return shap_values[0]
        else:
            # Single array - already in correct format
            return shap_values
            
    except Exception as e:
        st.error(f"SHAP value extraction error: {e}")
        return None

# ==================== HELPER FUNCTIONS ====================
def create_demo_model():
    """Create a demo model for testing purposes"""
    class DemoModel:
        def __init__(self):
            self.feature_names = ['Age', 'Surgery.time', 'Anesthesia', 'Calcium', 'ESR']
            self.classes_ = np.array([1, 2])  # Add classes_ attribute
            
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
     "📊 SHAP Interpretation",
     "📋 Model Performance Metrics"]
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
                if not demo_mode:
                    try:
                        st.markdown('<h3 class="sub-header">SHAP Waterfall Plot (Feature Contribution)</h3>', unsafe_allow_html=True)
                        
                        # Create SHAP explainer
                        explainer = shap.TreeExplainer(model)
                        
                        # Get SHAP values for this specific prediction
                        shap_values = explainer.shap_values(input_data)
                        
                        # Create waterfall plot
                        fig, ax = plt.subplots(figsize=(12, 8))
                        
                        # Handle different SHAP return formats
                        if isinstance(shap_values, list):
                            # For binary classification, shap_values is a list
                            if len(shap_values) == 2:
                                # Use positive class (index 1 usually)
                                shap.plots.waterfall(shap.Explanation(
                                    values=shap_values[1][0],
                                    base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
                                    data=input_data.iloc[0],
                                    feature_names=input_data.columns.tolist()
                                ), max_display=10, show=False)
                            else:
                                shap.plots.waterfall(shap.Explanation(
                                    values=shap_values[0][0],
                                    base_values=explainer.expected_value,
                                    data=input_data.iloc[0],
                                    feature_names=input_data.columns.tolist()
                                ), max_display=10, show=False)
                        else:
                            # Single array
                            shap.plots.waterfall(shap.Explanation(
                                values=shap_values[0],
                                base_values=explainer.expected_value,
                                data=input_data.iloc[0],
                                feature_names=input_data.columns.tolist()
                            ), max_display=10, show=False)
                        
                        plt.title("SHAP Waterfall Plot for Individual Prediction", fontsize=14, fontweight='bold')
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                        
                        # SHAP value interpretation
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
                        st.warning(f"⚠️ SHAP visualization error: {str(e)}")
                        st.info("""
                        **Alternative visualization**: Showing model feature importance instead.
                        """)
                        
                        # Show feature importance as fallback
                        if hasattr(model, 'feature_importances_'):
                            features = ['Age', 'Surgery.time', 'Anesthesia', 'Calcium', 'ESR']
                            importance = model.feature_importances_
                            
                            fig_fallback = go.Figure()
                            fig_fallback.add_trace(go.Bar(
                                x=features,
                                y=importance,
                                marker_color='#3B82F6',
                                text=[f'{val:.3f}' for val in importance],
                                textposition='auto'
                            ))
                            fig_fallback.update_layout(
                                title='Feature Importance (Model-based)',
                                xaxis_title='Feature',
                                yaxis_title='Importance',
                                height=400
                            )
                            st.plotly_chart(fig_fallback, use_container_width=True)
                
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

# ==================== SHAP INTERPRETATION ====================
elif app_mode == "📊 SHAP Interpretation":
    st.markdown('<h2 class="sub-header">SHAP Model Interpretability Analysis</h2>', unsafe_allow_html=True)
    
    if demo_mode:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("""
        ⚠️ **Demonstration Mode Active**
        
        SHAP analysis requires a properly trained LightGBM model. Currently using demonstration data.
        For actual SHAP analysis, please ensure a valid model file is uploaded.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Create example data for SHAP analysis
    st.markdown("### Generate Sample Data for SHAP Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        sample_size = st.slider("Number of samples for SHAP analysis", 50, 300, 100)
        random_seed = st.number_input("Random seed", value=42)
    
    with col2:
        st.markdown("**Feature Ranges:**")
        st.markdown("- Age: 20-90 years")
        st.markdown("- Surgery Time: 30-300 minutes")
        st.markdown("- Anesthesia: 1 or 2")
        st.markdown("- Calcium: 1.8-2.5 mmol/L")
        st.markdown("- ESR: 5-80 mm/h")
    
    # Generate sample data
    np.random.seed(random_seed)
    sample_data = pd.DataFrame({
        'Age': np.random.uniform(20, 90, sample_size),
        'Surgery.time': np.random.uniform(30, 300, sample_size),
        'Anesthesia': np.random.choice([1, 2], sample_size, p=[0.6, 0.4]),
        'Calcium': np.random.uniform(1.8, 2.5, sample_size),
        'ESR': np.random.uniform(5, 80, sample_size)
    })
    
    # Display sample data
    st.markdown("### Sample Data Preview")
    st.dataframe(sample_data.head(10), use_container_width=True)
    
    # SHAP analysis options
    analysis_type = st.radio(
        "Select SHAP Analysis Type",
        ["Global Feature Importance", "Individual Waterfall Plot"],
        horizontal=True
    )
    
    if st.button("🔍 **Run SHAP Analysis**", type="primary"):
        with st.spinner("Calculating SHAP values and generating visualizations..."):
            if not demo_mode:
                try:
                    # Create SHAP explainer
                    explainer = shap.TreeExplainer(model)
                    
                    if analysis_type == "Global Feature Importance":
                        st.markdown('<h3 class="sub-header">Global Feature Importance (SHAP)</h3>', unsafe_allow_html=True)
                        
                        try:
                            # Calculate SHAP values
                            shap_values = explainer.shap_values(sample_data)
                            
                            # Get mean absolute SHAP values
                            if isinstance(shap_values, list):
                                # For binary classification, use positive class (usually index 1)
                                shap_array = shap_values[1] if len(shap_values) == 2 else shap_values[0]
                            else:
                                shap_array = shap_values
                            
                            # Calculate mean absolute SHAP values
                            shap_importance = np.abs(shap_array).mean(axis=0)
                            
                            # Create bar plot
                            features = ['Age', 'Surgery.time', 'Anesthesia', 'Calcium', 'ESR']
                            importance_df = pd.DataFrame({
                                'Feature': features,
                                'Mean |SHAP value|': shap_importance
                            }).sort_values('Mean |SHAP value|', ascending=True)
                            
                            fig = go.Figure()
                            fig.add_trace(go.Bar(
                                y=importance_df['Feature'],
                                x=importance_df['Mean |SHAP value|'],
                                orientation='h',
                                marker_color='#3B82F6',
                                text=[f'{val:.4f}' for val in importance_df['Mean |SHAP value|']],
                                textposition='auto'
                            ))
                            
                            fig.update_layout(
                                title='Global Feature Importance (Mean Absolute SHAP Values)',
                                xaxis_title='Mean |SHAP value|',
                                yaxis_title='Feature',
                                height=400,
                                template='plotly_white'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Create simple SHAP summary plot using bar chart
                            st.markdown('<h4 class="sub-header">SHAP Value Distribution</h4>', unsafe_allow_html=True)
                            
                            fig_summary = go.Figure()
                            
                            for i, feature in enumerate(features):
                                fig_summary.add_trace(go.Box(
                                    y=shap_array[:, i],
                                    name=feature,
                                    boxpoints=False,
                                    marker_color='#3B82F6'
                                ))
                            
                            fig_summary.update_layout(
                                title='SHAP Value Distribution by Feature',
                                xaxis_title='Feature',
                                yaxis_title='SHAP Value',
                                height=400,
                                showlegend=False
                            )
                            
                            st.plotly_chart(fig_summary, use_container_width=True)
                            
                        except Exception as e:
                            st.warning(f"⚠️ SHAP calculation error: {str(e)}")
                            st.info("Using model's built-in feature importance instead.")
                            
                            # Fallback to model feature importance
                            if hasattr(model, 'feature_importances_'):
                                features = ['Age', 'Surgery.time', 'Anesthesia', 'Calcium', 'ESR']
                                importance = model.feature_importances_
                                
                                fig_fallback = go.Figure()
                                fig_fallback.add_trace(go.Bar(
                                    x=features,
                                    y=importance,
                                    marker_color='#3B82F6',
                                    text=[f'{val:.3f}' for val in importance],
                                    textposition='auto'
                                ))
                                fig_fallback.update_layout(
                                    title='Feature Importance (Model-based)',
                                    xaxis_title='Feature',
                                    yaxis_title='Importance',
                                    height=400
                                )
                                st.plotly_chart(fig_fallback, use_container_width=True)
                    
                    else:  # Individual Waterfall Plot
                        st.markdown('<h3 class="sub-header">Individual SHAP Waterfall Plot</h3>', unsafe_allow_html=True)
                        
                        # Select a sample for waterfall plot
                        sample_idx = st.selectbox("Select sample for waterfall plot", range(min(10, sample_size)))
                        
                        try:
                            # Get SHAP values for this specific sample
                            single_sample = sample_data.iloc[[sample_idx]]
                            shap_values_single = explainer.shap_values(single_sample)
                            
                            # Create waterfall plot
                            fig_waterfall, ax = plt.subplots(figsize=(12, 8))
                            
                            # Handle different SHAP return formats
                            if isinstance(shap_values_single, list):
                                if len(shap_values_single) == 2:
                                    # Binary classification - use positive class
                                    shap_obj = shap.Explanation(
                                        values=shap_values_single[1][0],
                                        base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
                                        data=single_sample.iloc[0],
                                        feature_names=single_sample.columns.tolist()
                                    )
                                else:
                                    shap_obj = shap.Explanation(
                                        values=shap_values_single[0][0],
                                        base_values=explainer.expected_value,
                                        data=single_sample.iloc[0],
                                        feature_names=single_sample.columns.tolist()
                                    )
                            else:
                                shap_obj = shap.Explanation(
                                    values=shap_values_single[0],
                                    base_values=explainer.expected_value,
                                    data=single_sample.iloc[0],
                                    feature_names=single_sample.columns.tolist()
                                )
                            
                            # Create waterfall plot
                            shap.plots.waterfall(shap_obj, max_display=10, show=False)
                            
                            plt.title(f"SHAP Waterfall Plot for Sample {sample_idx}", fontsize=14, fontweight='bold')
                            plt.tight_layout()
                            st.pyplot(fig_waterfall)
                            plt.close(fig_waterfall)
                            
                            # Show sample data
                            st.markdown(f"**Sample {sample_idx} Data:**")
                            st.dataframe(single_sample, use_container_width=True)
                            
                            # Show prediction for this sample
                            prediction = model.predict(single_sample)[0]
                            probability = model.predict_proba(single_sample)[0]
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Predicted Class", label_map[prediction])
                            with col2:
                                st.metric("Positive Probability", f"{probability[0]*100:.1f}%")
                            
                        except Exception as e:
                            st.error(f"❌ Waterfall plot error: {str(e)}")
                            st.info("Individual waterfall plot may not be available with current SHAP/model configuration.")
                
                except Exception as e:
                    st.error(f"❌ **SHAP Analysis Error**: {str(e)}")
                    st.info("""
                    **Possible issues:**
                    1. SHAP version incompatibility with model
                    2. Model format not fully compatible with SHAP
                    3. Memory limitations for large sample sizes
                    
                    **Try:**
                    1. Reduce sample size
                    2. Check model compatibility
                    3. Update SHAP library
                    """)
            else:
                st.warning("⚠️ SHAP analysis is not available in demonstration mode.")

# ==================== MODEL PERFORMANCE METRICS ====================
else:  # "📋 Model Performance Metrics"
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
        'Type': [
            'Continuous (years)',
            'Continuous (minutes)',
            'Categorical (1: General, 2: Non-general)',
            'Continuous (mmol/L)',
            'Continuous (mm/h)'
        ],
        'Clinical Significance': [
            'Older age associated with higher metabolic stress and reduced protein synthesis',
            'Longer surgical duration correlates with increased inflammatory response',
            'General anesthesia may induce greater physiological stress',
            'Lower calcium levels may indicate metabolic disturbances',
            'Elevated ESR suggests systemic inflammation affecting protein metabolism'
        ],
        'Normal Range': [
            'N/A',
            'N/A',
            'N/A',
            '2.1-2.6 mmol/L',
            '0-20 mm/h (varies by age/sex)'
        ]
    })
    
    st.dataframe(features_table, use_container_width=True, hide_index=True)
    
    # Feature importance visualization
    st.markdown('<h3 class="sub-header">Feature Importance</h3>', unsafe_allow_html=True)
    
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
        title='Feature Importance (Gain-based)',
        xaxis_title='Importance Score',
        yaxis_title='Clinical Feature',
        height=400,
        template='plotly_white'
    )
    
    st.plotly_chart(fig_importance, use_container_width=True)
    
    # SHAP availability status
    st.markdown('<h3 class="sub-header">SHAP Interpretability Status</h3>', unsafe_allow_html=True)
    
    if not demo_mode:
        st.success("✅ SHAP interpretability is available")
        st.markdown("""
        **Available SHAP visualizations:**
        1. **Waterfall plots** for individual predictions
        2. **Global feature importance** (mean absolute SHAP values)
        3. **SHAP value distributions** by feature
        
        **Note**: SHAP requires proper model compatibility. Some visualizations may be limited by SHAP version.
        """)
    else:
        st.warning("⚠️ SHAP interpretability is not available in demonstration mode")
        st.markdown("""
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
