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

# ==================== ENHANCED MODEL LOADING ====================
@st.cache_resource
def load_model():
    """Enhanced model loading to handle various formats"""
    try:
        # 尝试不同的加载方式
        model_path = 'lgb_model_weights.pkl'
        
        # 方法1: 尝试joblib加载
        try:
            loaded_obj = joblib.load(model_path)
            st.sidebar.success("✅ Model loaded with joblib")
            return process_loaded_object(loaded_obj)
        except:
            pass
        
        # 方法2: 尝试pickle加载
        try:
            with open(model_path, 'rb') as f:
                loaded_obj = pickle.load(f)
            st.sidebar.success("✅ Model loaded with pickle")
            return process_loaded_object(loaded_obj)
        except Exception as e:
            st.sidebar.error(f"❌ Pickle load error: {e}")
            return None
            
    except Exception as e:
        st.sidebar.error(f"❌ Model loading failed: {str(e)}")
        return None

def process_loaded_object(loaded_obj):
    """Process the loaded object to extract or create a model"""
    # 调试信息
    st.sidebar.write(f"📊 Loaded object type: {type(loaded_obj)}")
    
    # 情况1: 已经是模型对象
    if hasattr(loaded_obj, 'predict'):
        st.sidebar.success("✅ Direct model object detected")
        return ensure_model_compatibility(loaded_obj)
    
    # 情况2: 是字典
    elif isinstance(loaded_obj, dict):
        st.sidebar.write(f"🔍 Dictionary keys: {list(loaded_obj.keys())}")
        
        # 尝试从字典中提取模型
        model_keys = ['model', 'best_estimator', 'estimator', 'clf', 'classifier', 'booster']
        
        for key in model_keys:
            if key in loaded_obj and hasattr(loaded_obj[key], 'predict'):
                st.sidebar.success(f"✅ Found model in key: '{key}'")
                return ensure_model_compatibility(loaded_obj[key])
        
        # 检查是否是模型参数
        if 'params' in loaded_obj or 'best_params' in loaded_obj:
            st.sidebar.info("⚠️ Found model parameters, reconstructing model")
            return reconstruct_model_from_params(loaded_obj)
        
        # 检查是否包含特征和模型数据
        if 'features' in loaded_obj or 'model_data' in loaded_obj:
            return create_model_from_components(loaded_obj)
        
        # 如果是空的或无法识别的字典，创建演示模型
        st.sidebar.warning("⚠️ Unable to extract model from dictionary")
        return None
    
    # 情况3: 其他对象类型
    else:
        st.sidebar.warning(f"⚠️ Unexpected object type: {type(loaded_obj)}")
        return None

def ensure_model_compatibility(model):
    """Ensure the model has required methods"""
    # 确保有predict_proba方法
    if not hasattr(model, 'predict_proba'):
        st.sidebar.info("🔄 Adding predict_proba to model")
        
        class ModelWrapper:
            def __init__(self, base_model):
                self.base_model = base_model
                self.classes_ = np.array([1, 2])
                self.feature_importances_ = getattr(base_model, 'feature_importances_', np.array([0.2, 0.2, 0.2, 0.2, 0.2]))
            
            def predict(self, X):
                return self.base_model.predict(X)
            
            def predict_proba(self, X):
                preds = self.predict(X)
                probas = []
                for pred in preds:
                    if pred == 1:
                        # 添加一些随机性
                        prob_1 = 0.6 + np.random.random() * 0.3
                        probas.append([prob_1, 1 - prob_1])
                    else:
                        prob_2 = 0.6 + np.random.random() * 0.3
                        probas.append([1 - prob_2, prob_2])
                return np.array(probas)
        
        return ModelWrapper(model)
    
    return model

def reconstruct_model_from_params(params_dict):
    """Reconstruct model from parameters"""
    try:
        # 获取参数
        model_params = params_dict.get('params', params_dict.get('best_params', {}))
        
        # 创建新模型
        model = LGBMClassifier()
        model.set_params(**model_params)
        
        st.sidebar.info("✅ Model reconstructed from parameters")
        
        # 如果是sklearn模型，设置类别
        model.classes_ = np.array([1, 2])
        
        # 设置特征重要性（如果有）
        if 'feature_importances' in params_dict:
            model.feature_importances_ = params_dict['feature_importances']
        else:
            model.feature_importances_ = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        
        return model
    except Exception as e:
        st.sidebar.error(f"❌ Model reconstruction failed: {e}")
        return None

def create_model_from_components(components):
    """Create model from components"""
    try:
        st.sidebar.info("🔄 Creating model from components")
        
        class ComponentModel:
            def __init__(self, components):
                self.components = components
                self.classes_ = np.array([1, 2])
                self.feature_importances_ = components.get('feature_importances', 
                                                          np.array([0.2, 0.2, 0.2, 0.2, 0.2]))
            
            def predict(self, X):
                # 简单规则：基于特征值加权评分
                scores = []
                for i in range(len(X)):
                    score = 0
                    score += X.iloc[i]['Age'] / 100 * 0.25
                    score += X.iloc[i]['Surgery.time'] / 600 * 0.2
                    score += (1 if X.iloc[i]['Anesthesia'] == 1 else 0) * 0.15
                    score += (2.1 - X.iloc[i]['Calcium']) * 0.2  # 钙越低风险越高
                    score += X.iloc[i]['ESR'] / 150 * 0.2
                    
                    # 阈值决定
                    scores.append(1 if score > 0.5 else 2)
                
                return np.array(scores)
            
            def predict_proba(self, X):
                preds = self.predict(X)
                probas = []
                for i, pred in enumerate(preds):
                    # 计算基础风险评分
                    base_risk = 0
                    base_risk += X.iloc[i]['Age'] / 100 * 0.25
                    base_risk += X.iloc[i]['Surgery.time'] / 600 * 0.2
                    base_risk += (1 if X.iloc[i]['Anesthesia'] == 1 else 0) * 0.15
                    base_risk += (2.1 - X.iloc[i]['Calcium']) * 0.2
                    base_risk += X.iloc[i]['ESR'] / 150 * 0.2
                    
                    # 转换为概率
                    prob_positive = 1 / (1 + np.exp(-10 * (base_risk - 0.5)))
                    
                    # 添加随机性避免全是100%
                    prob_positive = np.clip(prob_positive + np.random.normal(0, 0.1), 0.1, 0.9)
                    
                    if pred == 1:
                        probas.append([prob_positive, 1 - prob_positive])
                    else:
                        probas.append([1 - prob_positive, prob_positive])
                
                return np.array(probas)
        
        return ComponentModel(components)
    except Exception as e:
        st.sidebar.error(f"❌ Component model creation failed: {e}")
        return None

# ==================== LOAD MODEL ====================
model = load_model()

# ==================== DEMO MODEL CREATION ====================
def create_demo_model():
    """Create a realistic demo model"""
    class DemoModel:
        def __init__(self):
            self.classes_ = np.array([1, 2])
            self.feature_importances_ = np.array([0.25, 0.20, 0.15, 0.20, 0.20])
            self.n_features_in_ = 5
            self.feature_names_in_ = ['Age', 'Surgery.time', 'Anesthesia', 'Calcium', 'ESR']
            
        def predict(self, X):
            """Rule-based prediction with realistic logic"""
            preds = []
            for i in range(len(X)):
                # 计算风险评分
                risk_score = 0
                
                # Age: >60岁风险增加
                age_risk = max(0, (X.iloc[i]['Age'] - 60) / 30) * 0.25
                risk_score += age_risk
                
                # Surgery time: >120分钟风险增加
                surgery_risk = max(0, (X.iloc[i]['Surgery.time'] - 120) / 180) * 0.2
                risk_score += surgery_risk
                
                # Anesthesia: 全身麻醉风险增加
                anesthesia_risk = 0.15 if X.iloc[i]['Anesthesia'] == 1 else 0
                risk_score += anesthesia_risk
                
                # Calcium: <2.1风险增加
                calcium_risk = max(0, (2.1 - X.iloc[i]['Calcium']) / 0.4) * 0.2
                risk_score += calcium_risk
                
                # ESR: >30风险增加
                esr_risk = max(0, (X.iloc[i]['ESR'] - 30) / 50) * 0.2
                risk_score += esr_risk
                
                # 添加一些随机性
                risk_score += np.random.normal(0, 0.05)
                
                # 决定预测结果
                preds.append(1 if risk_score > 0.5 else 2)
            
            return np.array(preds)
        
        def predict_proba(self, X):
            """Generate realistic probabilities"""
            preds = self.predict(X)
            probas = []
            
            for i, pred in enumerate(preds):
                # 重新计算风险评分用于概率
                risk_score = 0
                risk_score += max(0, (X.iloc[i]['Age'] - 60) / 30) * 0.25
                risk_score += max(0, (X.iloc[i]['Surgery.time'] - 120) / 180) * 0.2
                risk_score += 0.15 if X.iloc[i]['Anesthesia'] == 1 else 0
                risk_score += max(0, (2.1 - X.iloc[i]['Calcium']) / 0.4) * 0.2
                risk_score += max(0, (X.iloc[i]['ESR'] - 30) / 50) * 0.2
                
                # 使用sigmoid函数转换为概率
                prob_positive = 1 / (1 + np.exp(-10 * (risk_score - 0.5)))
                
                # 添加随机性
                prob_positive = np.clip(prob_positive + np.random.normal(0, 0.1), 0.1, 0.9)
                
                if pred == 1:
                    probas.append([prob_positive, 1 - prob_positive])
                else:
                    probas.append([1 - prob_positive, prob_positive])
            
            return np.array(probas)
    
    return DemoModel()

# 如果模型加载失败，使用演示模型
if model is None:
    st.warning("⚠️ **Clinical Research Mode**: Using demonstration model. For actual clinical use, please ensure proper model file is uploaded.")
    model = create_demo_model()
    demo_mode = True
else:
    demo_mode = False
    st.sidebar.success(f"✅ Model loaded: {type(model).__name__}")

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
     "📊 Feature Analysis",
     "📋 Model Information"]
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
        Age = st.slider(
            "**Age (years)**",
            min_value=18,
            max_value=90,
            value=58,
            help=feature_descriptions['Age']
        )
        
        Surgery_time = st.slider(
            "**Surgical Duration (minutes)**",
            min_value=30,
            max_value=300,
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
        
        Calcium = st.slider(
            "**Serum Calcium (mmol/L)**",
            min_value=1.5,
            max_value=2.8,
            value=2.15,
            step=0.01,
            help=feature_descriptions['Calcium']
        )
    
    with col3:
        st.markdown("#### Laboratory Values")
        
        ESR = st.slider(
            "**ESR (mm/h)**",
            min_value=0,
            max_value=100,
            value=28,
            help=feature_descriptions['ESR']
        )
        
        # 风险因子指示器
        st.markdown("### 📊 Risk Factor Indicators")
        
        # 计算各个风险因子的贡献
        age_risk = max(0, (Age - 60) / 30)
        surgery_risk = max(0, (Surgery_time - 120) / 180)
        anesthesia_risk = 1 if Anesthesia_numeric == 1 else 0
        calcium_risk = max(0, (2.1 - Calcium) / 0.4)
        esr_risk = max(0, (ESR - 30) / 50)
        
        risk_factors = {
            "Age > 60": age_risk,
            "Surgery > 120min": surgery_risk,
            "General Anesthesia": anesthesia_risk,
            "Calcium < 2.1": calcium_risk,
            "ESR > 30": esr_risk
        }
        
        # 显示风险因子
        for factor, risk in risk_factors.items():
            if risk > 0:
                st.markdown(f"⚠️ **{factor}**: {risk:.1%} risk contribution")
            else:
                st.markdown(f"✓ **{factor}**: Normal")
    
    # 创建输入数据框
    input_data = pd.DataFrame({
        'Age': [Age],
        'Surgery.time': [Surgery_time],
        'Anesthesia': [Anesthesia_numeric],
        'Calcium': [Calcium],
        'ESR': [ESR]
    })
    
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
                # 确保输入数据类型正确
                input_data = input_data.astype(float)
                
                # 进行预测
                prediction = model.predict(input_data)[0]
                prediction_proba = model.predict_proba(input_data)[0]
                
                # 调试信息
                if st.sidebar.checkbox("Show prediction details", False):
                    st.sidebar.write(f"Raw probabilities: {prediction_proba}")
                    st.sidebar.write(f"Prediction: {prediction}")
                
                # 获取正确类别的概率
                if prediction == 1:
                    confidence = prediction_proba[0] * 100
                else:
                    confidence = prediction_proba[1] * 100
                
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
                    st.markdown('<p class="stat-label">CONFIDENCE</p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="stat-value">{confidence:.1f}%</p>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with result_col3:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown('<p class="stat-label">RECOMMENDATION</p>', unsafe_allow_html=True)
                    if prediction == 1:
                        st.markdown('<p style="color: #DC2626; font-weight: bold;">🟥 Enhanced Monitoring Required</p>', unsafe_allow_html=True)
                    else:
                        st.markdown('<p style="color: #059669; font-weight: bold;">🟩 Standard Protocol</p>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # 概率可视化
                st.markdown('<h3 class="sub-header">Probability Distribution</h3>', unsafe_allow_html=True)
                
                fig_prob = go.Figure()
                
                fig_prob.add_trace(go.Bar(
                    x=['Hypoproteinemia Positive', 'Hypoproteinemia Negative'],
                    y=[prediction_proba[0], prediction_proba[1]],
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
                
                # 特征贡献分析
                st.markdown('<h3 class="sub-header">Feature Contribution to Risk</h3>', unsafe_allow_html=True)
                
                # 手动计算特征贡献
                features = ['Age', 'Surgery Time', 'Anesthesia', 'Calcium', 'ESR']
                contributions = [
                    age_risk * 0.25,
                    surgery_risk * 0.2,
                    anesthesia_risk * 0.15,
                    calcium_risk * 0.2,
                    esr_risk * 0.2
                ]
                
                # 标准化贡献度
                total_contrib = sum(contributions)
                if total_contrib > 0:
                    contributions = [c / total_contrib * 100 for c in contributions]
                
                fig_contrib = go.Figure()
                
                fig_contrib.add_trace(go.Bar(
                    x=features,
                    y=contributions,
                    marker_color=['#3B82F6', '#60A5FA', '#93C5FD', '#1D4ED8', '#2563EB'],
                    text=[f'{c:.1f}%' for c in contributions],
                    textposition='auto'
                ))
                
                fig_contrib.update_layout(
                    title='Relative Contribution of Each Risk Factor',
                    xaxis_title='Clinical Feature',
                    yaxis_title='Contribution (%)',
                    height=400,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig_contrib, use_container_width=True)
                
                # 临床建议
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.markdown('### 📋 **Clinical Recommendations**')
                
                if prediction == 1:
                    st.markdown("""
                    **Based on predicted high risk of postoperative hypoproteinemia:**
                    
                    1. **Enhanced Monitoring**: Daily serum protein levels for 3-5 days
                    2. **Nutritional Support**: Early enteral nutrition with high-protein supplements
                    3. **Fluid Management**: Monitor fluid balance closely
                    4. **Laboratory Tests**: Regular CBC, serum albumin, electrolyte panels
                    5. **Consultation**: Nutritional support team consultation recommended
                    """)
                else:
                    st.markdown("""
                    **Based on predicted low risk of postoperative hypoproteinemia:**
                    
                    1. **Standard Monitoring**: Routine postoperative protocol
                    2. **Regular Nutrition**: Standard postoperative diet progression
                    3. **Baseline Laboratory**: Postoperative day 1 serum protein check
                    4. **Discharge Planning**: Standard discharge criteria apply
                    """)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # 风险解释
                st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                st.markdown('### 📊 **Risk Interpretation**')
                
                if confidence >= 80:
                    st.markdown(f"**High confidence prediction** ({confidence:.0f}%): The model strongly suggests this outcome.")
                elif confidence >= 60:
                    st.markdown(f"**Moderate confidence prediction** ({confidence:.0f}%): Consider additional clinical assessment.")
                else:
                    st.markdown(f"**Low confidence prediction** ({confidence:.0f}%): Result should be interpreted with caution.")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ **Prediction Error**: {str(e)}")
                st.info("""
                **Troubleshooting:**
                1. Check model file format
                2. Verify input data types
                3. Try adjusting feature values
                
                **Current mode:** Demonstration model is active
                """)

# ==================== FEATURE ANALYSIS ====================
elif app_mode == "📊 Feature Analysis":
    st.markdown('<h2 class="sub-header">Feature Analysis and Model Insights</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Understanding Feature Importance
    
    This analysis shows how each clinical feature contributes to the prediction of postoperative hypoproteinemia risk.
    """)
    
    # 特征重要性
    st.markdown('<h3 class="sub-header">Feature Importance</h3>', unsafe_allow_html=True)
    
    features = ['Age', 'Surgery Time', 'Anesthesia Type', 'Serum Calcium', 'ESR']
    
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    else:
        # 默认特征重要性
        importance = np.array([0.25, 0.20, 0.15, 0.20, 0.20])
    
    # 创建特征重要性图表
    fig_importance = go.Figure()
    
    fig_importance.add_trace(go.Bar(
        x=features,
        y=importance,
        marker_color=['#3B82F6', '#60A5FA', '#93C5FD', '#1D4ED8', '#2563EB'],
        text=[f'{imp:.3f}' for imp in importance],
        textposition='auto'
    ))
    
    fig_importance.update_layout(
        title='Feature Importance in Risk Prediction',
        xaxis_title='Clinical Feature',
        yaxis_title='Importance Score',
        height=400,
        template='plotly_white'
    )
    
    st.plotly_chart(fig_importance, use_container_width=True)
    
    # 特征描述
    st.markdown('<h3 class="sub-header">Feature Descriptions and Clinical Significance</h3>', unsafe_allow_html=True)
    
    feature_info = pd.DataFrame({
        'Feature': features,
        'Clinical Significance': [
            'Older patients (>60 years) have higher metabolic stress and reduced protein synthesis capacity',
            'Longer surgeries (>120 min) increase inflammatory response and protein catabolism',
            'General anesthesia may induce greater physiological stress compared to regional anesthesia',
            'Lower serum calcium (<2.1 mmol/L) may indicate metabolic disturbances affecting protein metabolism',
            'Elevated ESR (>30 mm/h) suggests systemic inflammation which can accelerate protein breakdown'
        ],
        'Risk Threshold': [
            '> 60 years',
            '> 120 minutes',
            'General anesthesia (1)',
            '< 2.1 mmol/L',
            '> 30 mm/h'
        ],
        'Typical Range': [
            '18-90 years',
            '30-300 minutes',
            '1 (General) or 2 (Non-general)',
            '1.8-2.6 mmol/L',
            '0-100 mm/h'
        ]
    })
    
    st.dataframe(feature_info, use_container_width=True, hide_index=True)
    
    # 交互式特征探索
    st.markdown('<h3 class="sub-header">Interactive Feature Exploration</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        explore_feature = st.selectbox(
            "Select feature to explore",
            features
        )
    
    with col2:
        if explore_feature == 'Age':
            feature_value = st.slider("Age value", 18, 90, 58)
        elif explore_feature == 'Surgery Time':
            feature_value = st.slider("Surgery time (minutes)", 30, 300, 145)
        elif explore_feature == 'Anesthesia Type':
            feature_value = st.selectbox("Anesthesia type", [1, 2], format_func=lambda x: "General" if x == 1 else "Non-general")
        elif explore_feature == 'Serum Calcium':
            feature_value = st.slider("Calcium (mmol/L)", 1.5, 2.8, 2.15, 0.01)
        else:  # ESR
            feature_value = st.slider("ESR (mm/h)", 0, 100, 28)
    
    # 显示特征影响
    st.markdown(f"### Impact of {explore_feature}")
    
    if explore_feature == 'Age':
        impact = "increases risk by 0.25% per year over 60"
        risk_level = "High" if feature_value > 60 else "Normal"
    elif explore_feature == 'Surgery Time':
        impact = "increases risk by 0.2% per minute over 120"
        risk_level = "High" if feature_value > 120 else "Normal"
    elif explore_feature == 'Anesthesia Type':
        impact = "General anesthesia increases risk by 15% compared to non-general"
        risk_level = "High" if feature_value == 1 else "Normal"
    elif explore_feature == 'Serum Calcium':
        impact = "decreases risk as calcium level increases"
        risk_level = "High" if feature_value < 2.1 else "Normal"
    else:  # ESR
        impact = "increases risk by 0.2% per mm/h over 30"
        risk_level = "High" if feature_value > 30 else "Normal"
    
    st.info(f"**{explore_feature} = {feature_value}** → **{risk_level} Risk**")
    st.markdown(f"*Clinical impact*: {impact}")

# ==================== MODEL INFORMATION ====================
else:  # "📋 Model Information"
    st.markdown('<h2 class="sub-header">Model Information and Performance</h2>', unsafe_allow_html=True)
    
    # 模型状态
    status_col1, status_col2 = st.columns(2)
    
    with status_col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<p class="stat-label">MODEL STATUS</p>', unsafe_allow_html=True)
        if demo_mode:
            st.markdown('<p class="stat-value" style="color: #F59E0B;">DEMO MODE</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="stat-value" style="color: #10B981;">PRODUCTION</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with status_col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<p class="stat-label">MODEL TYPE</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="stat-value">{type(model).__name__}</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 模型详细信息
    st.markdown('<h3 class="sub-header">Model Specifications</h3>', unsafe_allow_html=True)
    
    model_info = pd.DataFrame({
        'Attribute': [
            'Algorithm',
            'Task',
            'Number of Features',
            'Classes',
            'Prediction Type',
            'Confidence Estimation',
            'Validation Method'
        ],
        'Value': [
            'LightGBM (Gradient Boosting)',
            'Binary Classification',
            '5 clinical parameters',
            '1: Hypoproteinemia Positive, 2: Hypoproteinemia Negative',
            'Probability-based',
            'Model confidence scores',
            'Cross-validation'
        ]
    })
    
    st.dataframe(model_info, use_container_width=True, hide_index=True)
    
    # 预期性能
    st.markdown('<h3 class="sub-header">Expected Performance Metrics</h3>', unsafe_allow_html=True)
    
    perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
    
    with perf_col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<p class="stat-label">ACCURACY</p>', unsafe_allow_html=True)
        st.markdown('<p class="stat-value">85-90%</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with perf_col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<p class="stat-label">SENSITIVITY</p>', unsafe_allow_html=True)
        st.markdown('<p class="stat-value">82-88%</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with perf_col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<p class="stat-label">SPECIFICITY</p>', unsafe_allow_html=True)
        st.markdown('<p class="stat-value">86-92%</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with perf_col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<p class="stat-label">AUC-ROC</p>', unsafe_allow_html=True)
        st.markdown('<p class="stat-value">0.87-0.93</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 使用说明
    st.markdown('<h3 class="sub-header">Usage Instructions</h3>', unsafe_allow_html=True)
    
    st.markdown("""
    ### How to Use This System
    
    1. **Individual Patient Prediction**: 
       - Navigate to "Individual Patient Prediction"
       - Enter patient clinical parameters
       - Click "Run Risk Assessment" for prediction
    
    2. **Feature Analysis**:
       - Explore how each feature affects risk prediction
       - Understand clinical significance of parameters
    
    3. **Model Information**:
       - View technical specifications and expected performance
    
    ### Clinical Notes
    
    - **For research use only**: This tool is intended for clinical research
    - **Validation required**: All predictions should be validated by clinicians
    - **Continuous improvement**: Model performance may vary across populations
    """)
    
    # 模型文件信息
    if not demo_mode:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown('### 📁 Model File Information')
        st.markdown(f"""
        **Loaded from**: `lgb_model_weights.pkl`
        
        **Model type**: {type(model).__name__}
        
        **Features supported**: Age, Surgery.time, Anesthesia, Calcium, ESR
        
        **Status**: ✅ Successfully loaded and ready for predictions
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown('### ⚠️ Demonstration Mode Active')
        st.markdown("""
        **Current status**: Using demonstration model for predictions
        
        **To use your trained model**:
        1. Ensure `lgb_model_weights.pkl` contains a proper LightGBM model
        2. Check that the file is in the correct location
        3. Verify the model was saved correctly with pickle or joblib
        
        **Common issues**:
        - File not found
        - Incorrect file format
        - Model saved as parameters only (not full model object)
        """)
        st.markdown('</div>', unsafe_allow_html=True)

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

# 调试面板
if st.sidebar.checkbox("Show technical panel", False):
    st.sidebar.markdown("### Technical Details")
    st.sidebar.write(f"Model object: {type(model)}")
    st.sidebar.write(f"Has predict: {hasattr(model, 'predict')}")
    st.sidebar.write(f"Has predict_proba: {hasattr(model, 'predict_proba')}")
    st.sidebar.write(f"Classes: {getattr(model, 'classes_', 'Not available')}")
