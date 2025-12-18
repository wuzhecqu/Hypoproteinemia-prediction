# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Hypoproteinemia Risk Predictor",
    page_icon="🏥",
    layout="wide"
)

# Title
st.title("🏥 Hypoproteinemia Risk Prediction after Surgery")
st.markdown("""
This tool predicts the risk of **Hypoproteinemia** in patients after surgery based on clinical features.  
The model is built with **LightGBM** and validated on clinical data.
""")

# Sidebar for navigation
st.sidebar.header("Navigation")
option = st.sidebar.selectbox(
    "Choose a page",
    ["📊 Single Patient Prediction", "📈 Model Interpretability", "📁 Validation Set Evaluation"]
)

# ====================== 1. Load Model from GitHub (or local) ======================
@st.cache_resource
def load_model():
    # Option 1: Load from local (for development)
    try:
        with open('lgb_model_weights.pkl', 'rb') as f:
            model_meta = pickle.load(f)
        st.sidebar.success("✅ Model loaded from local file")
    except:
        # Option 2: Load from GitHub raw URL
        github_url = "https://github.com/wuzhecqu/Hypoproteinemia-prediction/blob/main/lgb_model_weights.pkl"
        import requests
        response = requests.get(github_url)
        if response.status_code == 200:
            model_meta = pickle.loads(response.content)
            st.sidebar.success("✅ Model loaded from GitHub")
        else:
            st.error("❌ Model file not found. Please check the URL or local path.")
            st.stop()
    return model_meta

model_meta = load_model()
model = model_meta['model']
imputer = model_meta['imputer']
scaler = model_meta['scaler']
feature_cols = model_meta['feature_cols']
target_mapping = model_meta['target_mapping']
feature_descriptions = model_meta['feature_descriptions']

# ====================== 2. Single Patient Prediction ======================
if option == "📊 Single Patient Prediction":
    st.header("Single Patient Prediction")
    st.markdown("Enter patient clinical features below to predict Hypoproteinemia risk.")

    col1, col2 = st.columns(2)

    with col1:
        Age = st.number_input("Age (years)", min_value=0, max_value=120, value=60)
        Surgery_time = st.number_input("Surgery Duration (minutes)", min_value=0, max_value=600, value=120)
        Anesthesia = st.selectbox("Anesthesia Type", options=[1, 2], index=0,
                                  help="1: General, 2: Spinal, 3: Local")

    with col2:
        Calcium = st.number_input("Serum Calcium (mmol/L)", min_value=1.0, max_value=3.0, value=2.2, step=0.1)
        ESR = st.number_input("ESR (mm/h)", min_value=0, max_value=150, value=20)

    if st.button("Predict Risk", type="primary"):
        # Prepare input
        input_df = pd.DataFrame([[Age, Surgery_time, Anesthesia, Calcium, ESR]],
                                columns=feature_cols)
        
        # Preprocess
        input_imputed = imputer.transform(input_df)
        input_scaled = scaler.transform(input_imputed)
        
        # Predict
        prob = model.predict_proba(input_scaled)[0, 1]
        pred_class = model.predict(input_scaled)[0]
        
        # Display
        st.subheader("Prediction Result")
        
        risk_color = "red" if prob >= 0.5 else "green"
        st.markdown(f"**Predicted Risk Probability:** :{risk_color}[**{prob:.2%}**]")
        st.markdown(f"**Predicted Class:** **{target_mapping[pred_class]}**")
        
        # Risk gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Risk Level (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "green"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance for this prediction
        st.subheader("Feature Contribution for This Prediction")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_scaled)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # for class 1
        
        shap_df = pd.DataFrame({
            'Feature': feature_cols,
            'SHAP Value': shap_values[0],
            'Feature Value': input_df.iloc[0].values
        })
        shap_df['Impact'] = shap_df['SHAP Value'].apply(lambda x: "Increases Risk" if x > 0 else "Decreases Risk")
        
        fig2 = px.bar(shap_df, x='SHAP Value', y='Feature', orientation='h',
                     color='Impact', color_discrete_map={"Increases Risk": "red", "Decreases Risk": "green"},
                     title="SHAP Values for Current Prediction")
        st.plotly_chart(fig2, use_container_width=True)

# ====================== 3. Model Interpretability ======================
elif option == "📈 Model Interpretability":
    st.header("Model Interpretability Analysis")
    
    tab1, tab2, tab3 = st.tabs(["📊 Feature Importance", "📈 SHAP Summary", "🔍 Partial Dependence"])
    
    with tab1:
        st.subheader("Global Feature Importance")
        importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                     title="LightGBM Feature Importance")
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("**Feature Descriptions:**")
        for feat, desc in feature_descriptions.items():
            st.markdown(f"- **{feat}**: {desc}")
    
    with tab2:
        st.subheader("SHAP Summary Plot")
        st.info("Loading SHAP values may take a moment...")
        
        # Use validation data for SHAP (load from local)
        try:
            df_val = pd.read_excel('validation_data.xlsx')
            X_val_raw = df_val[feature_cols]
            X_val_imputed = imputer.transform(X_val_raw)
            X_val = scaler.transform(X_val_imputed)
            
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_val)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, X_val, feature_names=feature_cols, show=False)
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Could not generate SHAP plot: {e}")
            st.markdown("Please ensure `validation_data.xlsx` is in the same directory.")
    
    with tab3:
        st.subheader("Partial Dependence Plots")
        selected_feature = st.selectbox("Select a feature for PDP", feature_cols)
        
        # Simplified PDP
        try:
            df_val = pd.read_excel('validation_data.xlsx')
            X_val_raw = df_val[feature_cols]
            
            unique_vals = np.linspace(X_val_raw[selected_feature].min(), X_val_raw[selected_feature].max(), 50)
            pdp_vals = []
            
            for val in unique_vals:
                temp_df = X_val_raw.copy()
                temp_df[selected_feature] = val
                temp_imputed = imputer.transform(temp_df)
                temp_scaled = scaler.transform(temp_imputed)
                preds = model.predict_proba(temp_scaled)[:, 1]
                pdp_vals.append(preds.mean())
            
            pdp_df = pd.DataFrame({'Feature Value': unique_vals, 'Predicted Risk': pdp_vals})
            fig = px.line(pdp_df, x='Feature Value', y='Predicted Risk',
                         title=f"Partial Dependence for {selected_feature}")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"PDP not available: {e}")

# ====================== 4. Validation Set Evaluation ======================
elif option == "📁 Validation Set Evaluation":
    st.header("Validation Set Performance")
    
    try:
        df_val = pd.read_excel('validation_data.xlsx')
        df_val.columns = [col.strip() for col in df_val.columns]
        
        if 'Hypoproteinemia' in df_val.columns:
            df_val['Hypoproteinemia'] = df_val['Hypoproteinemia'].map({1: 1, 2: 0})
            y_true = df_val['Hypoproteinemia']
        else:
            st.error("Target column not found in validation data.")
            st.stop()
        
        X_val_raw = df_val[feature_cols]
        X_val_imputed = imputer.transform(X_val_raw)
        X_val = scaler.transform(X_val_imputed)
        
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]
        
        auc = roc_auc_score(y_true, y_prob)
        acc = accuracy_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred, zero_division=0)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("AUC Score", f"{auc:.3f}")
        with col2:
            st.metric("Accuracy", f"{acc:.3f}")
        with col3:
            st.metric("Recall (Sensitivity)", f"{recall:.3f}")
        
        # ROC Curve
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_df = pd.DataFrame({'FPR': fpr, 'TPR': tpr})
        fig = px.line(roc_df, x='FPR', y='TPR', title='ROC Curve')
        fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
        st.plotly_chart(fig, use_container_width=True)
        
        # Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        cm_df = pd.DataFrame(cm, 
                            columns=['Predicted Negative', 'Predicted Positive'],
                            index=['Actual Negative', 'Actual Positive'])
        st.subheader("Confusion Matrix")
        st.dataframe(cm_df.style.background_gradient(cmap='Blues'))
        
        # Show sample of validation data
        with st.expander("View Validation Data Sample"):
            st.dataframe(df_val.head(10))
            
    except Exception as e:
        st.error(f"Error loading validation data: {e}")
        st.markdown("Please ensure `validation_data.xlsx` is in the same directory.")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("**Clinical Decision Support Tool**\n\nFor research use only. Always consult with clinical professionals.")
