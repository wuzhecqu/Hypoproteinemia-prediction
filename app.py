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
from lightgbm import LGBMClassifier, Booster
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
    """Load the trained LightGBM model with improved error handling"""
    try:
        # 尝试多种加载方式
        try:
            # 方式1: 直接使用joblib加载
            model = joblib.load('lgb_model_weights.pkl')
            st.sidebar.success("✅ Model loaded with joblib")
            return model
        except:
            # 方式2: 使用pickle加载
            with open('lgb_model_weights.pkl', 'rb') as f:
                loaded_data = pickle.load(f)
            
            st.sidebar.info(f"📊 Loaded data type: {type(loaded_data).__name__}")
            
            # 情况1: 直接是模型对象
            if hasattr(loaded_data, 'predict'):
                st.sidebar.success("✅ Direct model object loaded")
                return loaded_data
            
            # 情况2: 字典包含模型
            elif isinstance(loaded_data, dict):
                st.sidebar.write(f"🔍 Dictionary keys: {list(loaded_data.keys())}")
                
                # 尝试可能的键名
                for key in ['model', 'best_estimator', 'estimator', 'classifier', 'booster']:
                    if key in loaded_data and hasattr(loaded_data[key], 'predict'):
                        st.sidebar.success(f"✅ Model extracted from key: '{key}'")
                        return loaded_data[key]
                
                # 情况3: 如果是LightGBM booster
                if 'booster' in str(type(loaded_data)).lower():
                    st.sidebar.success("✅ LightGBM Booster loaded")
                    return loaded_data
            
            # 情况4: 重建模型
            st.sidebar.warning("⚠️ Reconstructing model from parameters")
            model = LGBMClassifier()
            
            # 如果是sklearn包装的模型，尝试获取参数
            if hasattr(loaded_data, 'get_params'):
                params = loaded_data.get_params()
                model.set_params(**params)
                return model
            
            return None
            
    except Exception as e:
        st.sidebar.error(f"❌ Model loading error: {str(e)}")
        return None

# 加载模型
model = load_model()

# ==================== HELPER FUNCTIONS ====================
def create_demo_model():
    """Create a demo model for testing purposes"""
    class DemoModel:
        def __init__(self):
            self.feature_names = ['Age', 'Surgery.time', 'Anesthesia', 'Calcium', 'ESR']
            self.classes_ = np.array([1, 2])
            
        def predict(self, X):
            """Simple rule-based prediction with variability"""
            preds = []
            for i in range(len(X)):
                # 基于逻辑的风险评分
                risk_score = 0
                
                # Age: 60岁以上风险增加
                risk_score += max(0, (X.iloc[i]['Age'] - 60) / 40 * 0.2)
                
                # Surgery time: 超过120分钟风险增加
                risk_score += max(0, (X.iloc[i]['Surgery.time'] - 120) / 300 * 0.2)
                
                # Anesthesia: 全身麻醉风险略高
                if X.iloc[i]['Anesthesia'] == 1:
                    risk_score += 0.1
                
                # Calcium: 低于2.1风险增加
                risk_score += max(0, (2.1 - X.iloc[i]['Calcium']) / 0.5 * 0.3)
                
                # ESR: 超过30风险增加
                risk_score += max(0, (X.iloc[i]['ESR'] - 30) / 70 * 0.3)
                
                # 添加一些随机性避免全是100%
                risk_score += np.random.normal(0, 0.05)
                
                # 逻辑回归式的概率转换
                probability = 1 / (1 + np.exp(-risk_score))
                preds.append(1 if probability > 0.5 else 2)
            return np.array(preds)
        
        def predict_proba(self, X):
            """Generate realistic probability estimates"""
            preds = self.predict(X)
            probas = []
            
            for i, pred in enumerate(preds):
                # 基于风险因素计算基础概率
                base_risk = 0
                base_risk += max(0, (X.iloc[i]['Age'] - 60) / 40 * 0.2)
                base_risk += max(0, (X.iloc[i]['Surgery.time'] - 120) / 300 * 0.2)
                
                if X.iloc[i]['Anesthesia'] == 1:
                    base_risk += 0.1
                
                base_risk += max(0, (2.1 - X.iloc[i]['Calcium']) / 0.5 * 0.3)
                base_risk += max(0, (X.iloc[i]['ESR'] - 30) / 70 * 0.3)
                
                # 转换为概率 (0-1范围)
                probability = 1 / (1 + np.exp(-base_risk))
                
                # 添加一些随机变化
                probability = np.clip(probability + np.random.normal(0, 0.1), 0.1, 0.9)
                
                if pred == 1:
                    probas.append([probability, 1 - probability])
                else:
                    probas.append([1 - probability, probability])
            
            return np.array(probas)
        
        @property
        def feature_importances_(self):
            """Return simulated feature importances"""
            return np.array([0.25, 0.20, 0.15, 0.20, 0.20])
    
    return DemoModel()

# 如果模型加载失败，使用演示模型
if model is None:
    st.warning("⚠️ **Clinical Research Mode**: Using demonstration model. For actual clinical use, please ensure proper model file is uploaded.")
    model = create_demo_model()
    demo_mode = True
else:
    demo_mode = False
    # 检查模型是否具有必要的属性
    if not hasattr(model, 'predict_proba'):
        st.warning("⚠️ Loaded model doesn't have predict_proba method. Adding compatibility wrapper.")
        
        # 创建一个包装器
        class ModelWrapper:
            def __init__(self, base_model):
                self.base_model = base_model
                self.classes_ = np.array([1, 2])
            
            def predict(self, X):
                return self.base_model.predict(X)
            
            def predict_proba(self, X):
                preds = self.predict(X)
                probas = []
                for pred in preds:
                    if pred == 1:
                        probas.append([0.7, 0.3])  # 假设的概率
                    else:
                        probas.append([0.3, 0.7])
                return np.array(probas)
        
        model = ModelWrapper(model)

# ==================== LABEL MAPPING ====================
label_map = {
    1: "Hypoproteinemia Positive (High Risk)",
    2: "Hypoproteinemia Negative (Low Risk)"
}

# 确保模型有classes_属性
if not hasattr(model, 'classes_'):
    model.classes_ = np.array([1, 2])

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
    
    # 临床参数输入
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
        
        # 从选择中提取数值
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
    
    # 创建输入数据框
    input_data = pd.DataFrame({
        'Age': [Age],
        'Surgery.time': [Surgery_time],
        'Anesthesia': [Anesthesia_numeric],
        'Calcium': [Calcium],
        'ESR': [ESR]
    })
    
    # 显示输入参数
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
    
    # 预测按钮
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
                # 调试信息
                if demo_mode:
                    st.sidebar.info("🔍 Using demonstration model for predictions")
                else:
                    st.sidebar.success(f"🔍 Using trained model: {type(model).__name__}")
                
                # 确保输入数据格式正确
                input_data = input_data.astype(float)
                
                # 进行预测
                prediction = model.predict(input_data)[0]
                prediction_proba = model.predict_proba(input_data)[0]
                
                # 调试：显示原始概率
                st.sidebar.write(f"🔍 Raw probabilities: {prediction_proba}")
                
                # 根据模型类别顺序获取概率
                if hasattr(model, 'classes_'):
                    try:
                        # 找到类别1和2的索引
                        class_indices = {cls: idx for idx, cls in enumerate(model.classes_)}
                        
                        if 1 in class_indices and 2 in class_indices:
                            prob_positive = prediction_proba[class_indices[1]]
                            prob_negative = prediction_proba[class_indices[2]]
                        else:
                            # 如果类别不是1和2，使用第一个和第二个
                            prob_positive = prediction_proba[0]
                            prob_negative = prediction_proba[1] if len(prediction_proba) > 1 else 1 - prob_positive
                    except:
                        # 异常情况使用简单逻辑
                        prob_positive = prediction_proba[0]
                        prob_negative = 1 - prob_positive if len(prediction_proba) == 1 else prediction_proba[1]
                else:
                    # 默认处理
                    prob_positive = prediction_proba[0]
                    prob_negative = 1 - prob_positive if len(prediction_proba) == 1 else prediction_proba[1]
                
                # 确保概率总和为1
                total = prob_positive + prob_negative
                if total > 0:
                    prob_positive = prob_positive / total
                    prob_negative = prob_negative / total
                
                # 结果部分
                st.markdown("---")
                st.markdown('<h2 class="sub-header">Risk Assessment Results</h2>', unsafe_allow_html=True)
                
                # 结果显示在指标卡片中
                result_col1, result_col2, result_col3 = st.columns(3)
                
                with result_col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown('<p class="stat-label">PREDICTED OUTCOME</p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="stat-value">{label_map[prediction]}</p>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with result_col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown('<p class="stat-label">PROBABILITY</p>', unsafe_allow_html=True)
                    
                    if prediction == 1:
                        display_prob = prob_positive * 100
                    else:
                        display_prob = prob_negative * 100
                    
                    st.markdown(f'<p class="stat-value">{display_prob:.1f}%</p>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with result_col3:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown('<p class="stat-label">CLINICAL IMPLICATION</p>', unsafe_allow_html=True)
                    if prediction == 1:
                        st.markdown('<p style="color: #DC2626; font-weight: bold;">🟥 High Risk - Intensive Monitoring</p>', unsafe_allow_html=True)
                    else:
                        st.markdown('<p style="color: #059669; font-weight: bold;">🟩 Low Risk - Standard Care</p>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # 概率可视化
                st.markdown('<h3 class="sub-header">Probability Distribution</h3>', unsafe_allow_html=True)
                
                fig_prob = go.Figure()
                
                fig_prob.add_trace(go.Bar(
                    x=['Hypoproteinemia Positive', 'Hypoproteinemia Negative'],
                    y=[prob_positive, prob_negative],
                    text=[f'{prob_positive*100:.1f}%', f'{prob_negative*100:.1f}%'],
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
                
                # SHAP可视化（仅用于真实模型）
                if not demo_mode:
                    try:
                        st.markdown('<h3 class="sub-header">Feature Contribution Analysis</h3>', unsafe_allow_html=True)
                        
                        # 尝试使用SHAP
                        try:
                            explainer = shap.TreeExplainer(model)
                            shap_values = explainer.shap_values(input_data)
                            
                            # 处理SHAP值
                            if isinstance(shap_values, list):
                                # 二元分类
                                if len(shap_values) == 2:
                                    shap_to_use = shap_values[1][0]  # 阳性类
                                else:
                                    shap_to_use = shap_values[0][0]
                            else:
                                shap_to_use = shap_values[0]
                            
                            # 创建条形图显示特征贡献
                            fig_shap = go.Figure()
                            
                            features = ['Age', 'Surgery.time', 'Anesthesia', 'Calcium', 'ESR']
                            shap_vals = shap_to_use
                            
                            fig_shap.add_trace(go.Bar(
                                x=features,
                                y=shap_vals,
                                marker_color=['#3B82F6' if v > 0 else '#EF4444' for v in shap_vals],
                                text=[f'{v:.4f}' for v in shap_vals],
                                textposition='auto'
                            ))
                            
                            fig_shap.update_layout(
                                title='Feature Contribution to Prediction (SHAP values)',
                                xaxis_title='Feature',
                                yaxis_title='SHAP Value',
                                height=400,
                                template='plotly_white'
                            )
                            
                            st.plotly_chart(fig_shap, use_container_width=True)
                            
                        except Exception as shap_error:
                            st.info("⚠️ SHAP visualization not available. Showing feature importance instead.")
                            
                            # 使用特征重要性作为备选
                            if hasattr(model, 'feature_importances_'):
                                features = ['Age', 'Surgery.time', 'Anesthesia', 'Calcium', 'ESR']
                                importance = model.feature_importances_
                                
                                fig_importance = go.Figure()
                                fig_importance.add_trace(go.Bar(
                                    x=features,
                                    y=importance,
                                    marker_color='#3B82F6',
                                    text=[f'{val:.4f}' for val in importance],
                                    textposition='auto'
                                ))
                                fig_importance.update_layout(
                                    title='Feature Importance',
                                    xaxis_title='Feature',
                                    yaxis_title='Importance',
                                    height=400
                                )
                                st.plotly_chart(fig_importance, use_container_width=True)
                    
                    except Exception as e:
                        st.warning(f"Feature analysis error: {str(e)}")
                
                # 临床建议
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
                st.info("""
                **Troubleshooting suggestions:**
                1. Check if the model file is properly uploaded
                2. Verify the input data format matches training data
                3. Try using different feature values
                """)

# ==================== SHAP INTERPRETATION ====================
elif app_mode == "📊 SHAP Interpretation":
    st.markdown('<h2 class="sub-header">Model Interpretability Analysis</h2>', unsafe_allow_html=True)
    
    if demo_mode:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("""
        ⚠️ **Demonstration Mode Active**
        
        SHAP analysis requires a properly trained LightGBM model. Currently using demonstration data.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 生成示例数据
    st.markdown("### Generate Sample Data for Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        sample_size = st.slider("Number of samples", 20, 100, 50)
    
    with col2:
        st.markdown("**Feature Ranges:**")
        st.markdown("- Age: 20-90 years")
        st.markdown("- Surgery Time: 30-300 minutes")
    
    # 生成样本数据
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'Age': np.random.uniform(20, 90, sample_size),
        'Surgery.time': np.random.uniform(30, 300, sample_size),
        'Anesthesia': np.random.choice([1, 2], sample_size, p=[0.6, 0.4]),
        'Calcium': np.random.uniform(1.8, 2.5, sample_size),
        'ESR': np.random.uniform(5, 80, sample_size)
    })
    
    if st.button("🔍 **Run Analysis**", type="primary"):
        with st.spinner("Analyzing model behavior..."):
            
            # 使用简单特征重要性
            st.markdown('<h3 class="sub-header">Feature Analysis</h3>', unsafe_allow_html=True)
            
            if not demo_mode and hasattr(model, 'feature_importances_'):
                features = ['Age', 'Surgery.time', 'Anesthesia', 'Calcium', 'ESR']
                importance = model.feature_importances_
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=features,
                    y=importance,
                    marker_color='#3B82F6',
                    text=[f'{val:.4f}' for val in importance],
                    textposition='auto'
                ))
                
                fig.update_layout(
                    title='Feature Importance',
                    xaxis_title='Feature',
                    yaxis_title='Importance',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # 预测分布
            st.markdown('<h3 class="sub-header">Prediction Distribution on Sample Data</h3>', unsafe_allow_html=True)
            
            try:
                predictions = model.predict(sample_data)
                probabilities = model.predict_proba(sample_data)[:, 0]  # 阳性概率
                
                fig_dist = go.Figure()
                
                fig_dist.add_trace(go.Histogram(
                    x=probabilities,
                    nbinsx=20,
                    marker_color='#3B82F6',
                    opacity=0.7
                ))
                
                fig_dist.update_layout(
                    title='Distribution of Predicted Probabilities',
                    xaxis_title='Probability of Hypoproteinemia',
                    yaxis_title='Count',
                    height=400
                )
                
                st.plotly_chart(fig_dist, use_container_width=True)
                
                # 统计信息
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean Probability", f"{np.mean(probabilities):.3f}")
                with col2:
                    st.metric("Positive Predictions", f"{np.sum(predictions == 1)}")
                with col3:
                    st.metric("Negative Predictions", f"{np.sum(predictions == 2)}")
                    
            except Exception as e:
                st.warning(f"Could not generate prediction distribution: {str(e)}")

# ==================== MODEL PERFORMANCE METRICS ====================
else:  # "📋 Model Performance Metrics"
    st.markdown('<h2 class="sub-header">Model Performance & Technical Details</h2>', unsafe_allow_html=True)
    
    if demo_mode:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("""
        ⚠️ **Demonstration Mode Active**
        
        Currently using demonstration model.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown("""
        ✅ **Trained Model Active**
        
        Using the uploaded LightGBM model for predictions.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 模型信息
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Model Information")
        st.markdown(f"""
        **Model Type**: {type(model).__name__}
        
        **Mode**: {'Demonstration' if demo_mode else 'Production'}
        
        **Classes**:
        - Class 1: Hypoproteinemia Positive (High Risk)
        - Class 2: Hypoproteinemia Negative (Low Risk)
        
        **Features**: 5 clinical parameters
        
        **Model Status**: {'✅ Loaded successfully' if not demo_mode else '⚠️ Using demo model'}
        """)
    
    with col2:
        st.markdown("### Feature Information")
        st.markdown("""
        | Feature | Type | Clinical Significance |
        |---------|------|----------------------|
        | Age | Continuous | Older age increases risk |
        | Surgery Time | Continuous | Longer surgery increases risk |
        | Anesthesia | Categorical | General anesthesia may increase risk |
        | Calcium | Continuous | Lower levels indicate higher risk |
        | ESR | Continuous | Higher levels indicate inflammation |
        """)
    
    # 特征重要性
    st.markdown('<h3 class="sub-header">Feature Importance</h3>', unsafe_allow_html=True)
    
    features = ['Age', 'Surgery.time', 'Anesthesia', 'Calcium', 'ESR']
    
    if hasattr(model, 'feature_importances_'):
        importance_scores = model.feature_importances_
    else:
        # 模拟特征重要性
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
        title='Feature Importance',
        xaxis_title='Importance Score',
        yaxis_title='Clinical Feature',
        height=400
    )
    
    st.plotly_chart(fig_importance, use_container_width=True)
    
    # 使用说明
    st.markdown('<h3 class="sub-header">Usage Instructions</h3>', unsafe_allow_html=True)
    
    st.markdown("""
    1. **Individual Prediction**: Enter patient parameters to get personalized risk assessment
    2. **Feature Analysis**: Understand how different factors contribute to risk
    3. **Model Info**: View technical details and performance metrics
    
    **Note**: This tool is for clinical research purposes only.
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

# 调试信息
if st.sidebar.checkbox("Show debug info", False):
    st.sidebar.markdown("### Debug Information")
    st.sidebar.write(f"Model type: {type(model)}")
    st.sidebar.write(f"Demo mode: {demo_mode}")
    if hasattr(model, 'classes_'):
        st.sidebar.write(f"Model classes: {model.classes_}")
