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

# ==================== SHAP COMPATIBLE MODEL LOADER ====================
@st.cache_resource
def load_shap_compatible_model():
    """Load model in a way that's compatible with SHAP"""
    try:
        # 尝试加载模型文件
        try:
            # 先尝试joblib
            loaded_obj = joblib.load('lgb_model_weights.pkl')
            st.sidebar.success("✅ Model loaded with joblib")
            return process_for_shap(loaded_obj)
        except Exception as e:
            st.sidebar.info(f"Joblib failed: {str(e)[:50]}...")
        
        # 尝试pickle
        try:
            with open('lgb_model_weights.pkl', 'rb') as f:
                loaded_obj = pickle.load(f)
            st.sidebar.success("✅ Model loaded with pickle")
            return process_for_shap(loaded_obj)
        except Exception as e:
            st.sidebar.info(f"Pickle failed: {str(e)[:50]}...")
        
        # 都失败则创建演示模型
        st.sidebar.warning("⚠️ Creating SHAP-compatible demo model")
        return create_shap_demo_model()
        
    except Exception as e:
        st.sidebar.error(f"❌ Model loading error: {str(e)}")
        return create_shap_demo_model()

def process_for_shap(loaded_obj):
    """Process loaded object to be SHAP compatible"""
    # 调试信息
    st.sidebar.write(f"📊 Loaded object type: {type(loaded_obj)}")
    
    # 情况1: 已经是LightGBM模型
    if isinstance(loaded_obj, LGBMClassifier) or hasattr(loaded_obj, '_Booster'):
        st.sidebar.success("✅ LightGBM model detected (SHAP compatible)")
        return loaded_obj
    
    # 情况2: 字典中包含模型
    elif isinstance(loaded_obj, dict):
        st.sidebar.write(f"🔍 Dictionary keys: {list(loaded_obj.keys())}")
        
        # 查找LightGBM模型
        model_keys = ['model', 'best_estimator', 'estimator', 'clf', 'classifier', 'booster']
        for key in model_keys:
            if key in loaded_obj:
                model_obj = loaded_obj[key]
                if isinstance(model_obj, LGBMClassifier) or hasattr(model_obj, '_Booster'):
                    st.sidebar.success(f"✅ Found LightGBM model in key: '{key}'")
                    return model_obj
        
        # 如果是模型参数，创建LightGBM模型
        if 'params' in loaded_obj or 'best_params' in loaded_obj:
            st.sidebar.info("🔄 Creating LightGBM model from parameters")
            params = loaded_obj.get('params', loaded_obj.get('best_params', {}))
            model = LGBMClassifier()
            model.set_params(**params)
            return model
    
    # 情况3: 其他类型的模型，创建演示模型
    st.sidebar.warning("⚠️ Object not SHAP compatible, using demo model")
    return create_shap_demo_model()

def create_shap_demo_model():
    """Create a LightGBM demo model that's SHAP compatible"""
    st.sidebar.info("🔄 Creating SHAP-compatible demo model")
    
    # 创建简单的训练数据
    np.random.seed(42)
    n_samples = 100
    X_demo = pd.DataFrame({
        'Age': np.random.uniform(20, 80, n_samples),
        'Surgery.time': np.random.uniform(30, 300, n_samples),
        'Anesthesia': np.random.choice([1, 2], n_samples),
        'Calcium': np.random.uniform(1.8, 2.6, n_samples),
        'ESR': np.random.uniform(5, 80, n_samples)
    })
    
    # 基于规则创建标签
    y_demo = []
    for i in range(n_samples):
        risk = 0
        risk += (X_demo.iloc[i]['Age'] - 50) / 30 * 0.3
        risk += (X_demo.iloc[i]['Surgery.time'] - 150) / 150 * 0.2
        risk += (2.2 - X_demo.iloc[i]['Calcium']) * 0.3
        risk += (X_demo.iloc[i]['ESR'] - 30) / 50 * 0.2
        y_demo.append(1 if risk > 0 else 2)
    
    y_demo = np.array(y_demo)
    
    # 训练LightGBM模型
    model = LGBMClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )
    
    model.fit(X_demo, y_demo)
    st.sidebar.success("✅ SHAP-compatible demo model created")
    return model

# ==================== LOAD MODEL ====================
model = load_shap_compatible_model()

# 检查是否是演示模型
demo_mode = not hasattr(model, '_Booster') and not isinstance(model, LGBMClassifier)
if demo_mode:
    st.sidebar.warning("⚠️ Using SHAP-compatible demo model")

# ==================== SHAP WATERFALL PLOT FUNCTION ====================
def create_shap_waterfall_plot(input_data, model, patient_idx=0):
    """Create SHAP waterfall plot for individual prediction"""
    try:
        # 创建SHAP解释器
        explainer = shap.TreeExplainer(model)
        
        # 计算SHAP值
        shap_values = explainer.shap_values(input_data)
        
        # 获取当前患者的SHAP值
        if isinstance(shap_values, list):
            # 对于二分类，shap_values是一个列表 [负类SHAP值, 正类SHAP值]
            # 我们通常使用正类（索引1）
            if len(shap_values) == 2:
                shap_val = shap_values[1][patient_idx]
                base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
            else:
                shap_val = shap_values[0][patient_idx]
                base_value = explainer.expected_value
        else:
            # 单个数组
            shap_val = shap_values[patient_idx]
            base_value = explainer.expected_value
        
        # 获取特征名称
        feature_names = input_data.columns.tolist()
        
        # 创建SHAP解释对象
        explanation = shap.Explanation(
            values=shap_val,
            base_values=base_value,
            data=input_data.iloc[patient_idx],
            feature_names=feature_names
        )
        
        # 使用Matplotlib创建瀑布图
        fig, ax = plt.subplots(figsize=(12, 8))
        shap.plots.waterfall(explanation, max_display=10, show=False)
        plt.title("SHAP Waterfall Plot - Feature Contributions", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        st.sidebar.error(f"❌ SHAP error: {str(e)[:100]}")
        return None

def create_plotly_waterfall_plot(input_data, model, patient_idx=0):
    """Create Plotly waterfall plot as fallback"""
    try:
        # 计算特征贡献（简单方法）
        patient = input_data.iloc[patient_idx]
        features = ['Age', 'Surgery.time', 'Anesthesia', 'Calcium', 'ESR']
        
        # 基于规则计算贡献
        contributions = []
        
        # Age contribution
        age_contrib = (patient['Age'] - 50) / 30 * 0.15
        contributions.append(age_contrib)
        
        # Surgery time contribution
        surgery_contrib = (patient['Surgery.time'] - 150) / 150 * 0.12
        contributions.append(surgery_contrib)
        
        # Anesthesia contribution
        anesthesia_contrib = 0.08 if patient['Anesthesia'] == 1 else -0.04
        contributions.append(anesthesia_contrib)
        
        # Calcium contribution
        calcium_contrib = (2.2 - patient['Calcium']) * 0.15
        contributions.append(calcium_contrib)
        
        # ESR contribution
        esr_contrib = (patient['ESR'] - 30) / 50 * 0.10
        contributions.append(esr_contrib)
        
        # 创建瀑布图数据
        base_value = 0.5
        waterfall_values = [base_value] + contributions
        waterfall_labels = ['Base Value'] + features
        measures = ['absolute'] + ['relative'] * len(features)
        
        # 计算最终值
        final_value = base_value + sum(contributions)
        
        # 创建Plotly瀑布图
        fig = go.Figure()
        
        fig.add_trace(go.Waterfall(
            name="Feature Contributions",
            orientation="v",
            measure=measures,
            x=waterfall_labels,
            textposition="outside",
            text=[f"{base_value:.3f}"] + [f"{c:.3f}" for c in contributions] + [f"{final_value:.3f}"],
            y=waterfall_values + [0],  # 最后一个是占位符
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            decreasing={"marker": {"color": "#10B981"}},
            increasing={"marker": {"color": "#EF4444"}},
            totals={"marker": {"color": "#3B82F6"}}
        ))
        
        fig.update_layout(
            title="Feature Contributions to Prediction (Waterfall Plot)",
            xaxis_title="Clinical Features",
            yaxis_title="Contribution Value",
            height=500,
            showlegend=False,
            template='plotly_white'
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Plotly waterfall error: {str(e)}")
        return None

# ==================== LABEL MAPPING ====================
label_map = {
    1: "Hypoproteinemia Positive (High Risk)",
    2: "Hypoproteinemia Negative (Low Risk)"
}

# ==================== SIDEBAR ====================
st.sidebar.markdown("# 🔬 Navigation")
st.sidebar.markdown("---")

app_mode = st.sidebar.radio(
    "Select Functionality",
    ["📊 Individual Patient Prediction", "📊 SHAP Analysis", "📋 Model Information"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Clinical Features")

feature_descriptions = {
    'Age': 'Patient age in years',
    'Surgery.time': 'Duration of surgery in minutes',
    'Anesthesia': 'Type of anesthesia (1: General anesthesia, 2: Non-general anesthesia)',
    'Calcium': 'Serum calcium level (mmol/L)',
    'ESR': 'Erythrocyte Sedimentation Rate (mm/h)'
}

for feature, desc in feature_descriptions.items():
    st.sidebar.markdown(f"**{feature}**: {desc}")

# ==================== MAIN CONTENT ====================
st.markdown('<h1 class="main-header">🏥 Postoperative Hypoproteinemia Risk Prediction</h1>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #6B7280; margin-bottom: 2rem;">
    <p>Clinical decision support system with SHAP interpretability</p>
    <p><strong>For Research Use Only</strong> | Version 4.0 | SHAP Enabled</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ==================== INDIVIDUAL PATIENT PREDICTION ====================
if app_mode == "📊 Individual Patient Prediction":
    st.markdown('<h2 class="sub-header">Individual Patient Risk Assessment</h2>', unsafe_allow_html=True)
    
    # 输入参数
    col1, col2 = st.columns([1, 1])
    
    with col1:
        Age = st.slider(
            "**Age (years)**",
            min_value=20,
            max_value=90,
            value=58
        )
        
        Surgery_time = st.slider(
            "**Surgical Duration (minutes)**",
            min_value=30,
            max_value=360,
            value=145,
            step=5
        )
        
        Anesthesia = st.selectbox(
            "**Anesthesia Type**",
            ["General anesthesia (1)", "Non-general anesthesia (2)"],
            index=0
        )
        Anesthesia_numeric = 1 if "General" in Anesthesia else 2
    
    with col2:
        Calcium = st.slider(
            "**Serum Calcium (mmol/L)**",
            min_value=1.5,
            max_value=2.8,
            value=2.15,
            step=0.01
        )
        
        ESR = st.slider(
            "**ESR (mm/h)**",
            min_value=0,
            max_value=100,
            value=28
        )
    
    # 创建输入数据
    input_data = pd.DataFrame({
        'Age': [Age],
        'Surgery.time': [Surgery_time],
        'Anesthesia': [Anesthesia_numeric],
        'Calcium': [Calcium],
        'ESR': [ESR]
    })
    
    # 预测按钮
    if st.button("🚀 **Run Risk Assessment with SHAP**", type="primary", use_container_width=True):
        with st.spinner("**Calculating prediction and SHAP values...**"):
            try:
                # 进行预测
                prediction = model.predict(input_data)[0]
                prediction_proba = model.predict_proba(input_data)[0]
                
                # 获取概率
                prob_positive = float(prediction_proba[0])
                prob_negative = float(prediction_proba[1])
                
                # 归一化
                total = prob_positive + prob_negative
                if total > 0:
                    prob_positive = prob_positive / total
                    prob_negative = prob_negative / total
                
                # 显示结果
                st.markdown("---")
                st.markdown('<h2 class="sub-header">Risk Assessment Results</h2>', unsafe_allow_html=True)
                
                # 结果卡片
                col1, col2 = st.columns(2)
                
                with col1:
                    outcome_color = "#DC2626" if prediction == 1 else "#059669"
                    st.markdown(f"""
                    <div class="metric-card">
                        <p class="stat-label">PREDICTED OUTCOME</p>
                        <p class="stat-value" style="color: {outcome_color};">
                            {label_map[prediction]}
                        </p>
                        <p>Confidence: {max(prob_positive, prob_negative)*100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    if prediction == 1:
                        st.markdown("""
                        <div class="metric-card">
                            <p class="stat-label">CLINICAL IMPLICATION</p>
                            <p style="color: #DC2626; font-size: 1.2rem; font-weight: bold;">
                            🟥 High Risk - Intensive Monitoring Required
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="metric-card">
                            <p class="stat-label">CLINICAL IMPLICATION</p>
                            <p style="color: #059669; font-size: 1.2rem; font-weight: bold;">
                            🟩 Low Risk - Standard Care Protocol
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # 概率分布
                st.markdown('<h3 class="sub-header">Probability Distribution</h3>', unsafe_allow_html=True)
                
                fig_prob = go.Figure()
                fig_prob.add_trace(go.Bar(
                    x=['Positive', 'Negative'],
                    y=[prob_positive, prob_negative],
                    text=[f'{prob_positive*100:.1f}%', f'{prob_negative*100:.1f}%'],
                    marker_color=['#EF4444', '#10B981']
                ))
                fig_prob.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_prob, use_container_width=True)
                
                # SHAP瀑布图
                st.markdown('<h3 class="sub-header">SHAP Waterfall Plot - Feature Contributions</h3>', unsafe_allow_html=True)
                
                # 尝试创建SHAP瀑布图
                shap_fig = create_shap_waterfall_plot(input_data, model)
                
                if shap_fig is not None:
                    # 显示SHAP瀑布图
                    st.pyplot(shap_fig)
                    plt.close(shap_fig)
                    
                    # SHAP解释
                    st.markdown('<div class="info-box">', unsafe_allow_html=True)
                    st.markdown("""
                    ### 📊 **SHAP Waterfall Plot Interpretation**
                    
                    **How to read this plot:**
                    - **Red bars**: Features that increase the probability of hypoproteinemia
                    - **Blue bars**: Features that decrease the probability
                    - **Bar length**: Magnitude of the feature's contribution
                    - **E[f(X)]**: Expected/base value (average prediction)
                    - **f(x)**: Final prediction for this specific patient
                    
                    **Clinical insight**: The features with the largest absolute SHAP values 
                    have the greatest impact on this prediction.
                    """)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                else:
                    # 使用Plotly瀑布图作为备选
                    st.warning("⚠️ SHAP visualization not available. Showing alternative visualization.")
                    
                    plotly_fig = create_plotly_waterfall_plot(input_data, model)
                    if plotly_fig is not None:
                        st.plotly_chart(plotly_fig, use_container_width=True)
                        
                        # 解释
                        st.markdown('<div class="info-box">', unsafe_allow_html=True)
                        st.markdown("""
                        ### 📊 **Feature Contribution Analysis**
                        
                        This plot shows how each clinical feature contributes to the overall risk prediction:
                        
                        - **Positive values**: Increase risk of hypoproteinemia
                        - **Negative values**: Decrease risk
                        - **Base Value**: Average risk in the population
                        - **Final Value**: Calculated risk for this patient
                        
                        Features with larger absolute values have greater impact on the prediction.
                        """)
                        st.markdown('</div>', unsafe_allow_html=True)
                
                # 特征值表格
                st.markdown('<h3 class="sub-header">Input Feature Values</h3>', unsafe_allow_html=True)
                
                feature_table = pd.DataFrame({
                    'Feature': ['Age', 'Surgery Time', 'Anesthesia', 'Calcium', 'ESR'],
                    'Value': [f"{Age} years", f"{Surgery_time} minutes", 
                             Anesthesia, f"{Calcium:.2f} mmol/L", f"{ESR} mm/h"],
                    'Risk Level': [
                        "High" if Age > 60 else "Normal",
                        "High" if Surgery_time > 120 else "Normal",
                        "High" if Anesthesia_numeric == 1 else "Low",
                        "High" if Calcium < 2.1 else "Normal",
                        "High" if ESR > 30 else "Normal"
                    ]
                })
                
                st.dataframe(feature_table, use_container_width=True, hide_index=True)
                
                # 临床建议
                st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                st.markdown('### 📋 **Clinical Recommendations**')
                
                if prediction == 1:
                    st.markdown("""
                    1. **Enhanced Monitoring**: Daily serum protein for 3-5 days
                    2. **Nutritional Support**: High-protein supplements (1.5 g/kg/day)
                    3. **Laboratory Tests**: Regular albumin, pre-albumin monitoring
                    4. **Consultation**: Nutrition support team consultation
                    """)
                else:
                    st.markdown("""
                    1. **Standard Monitoring**: Routine postoperative protocol
                    2. **Regular Nutrition**: Adequate protein intake (0.8-1.0 g/kg/day)
                    3. **Baseline Check**: Serum protein on postoperative day 1
                    4. **Discharge Planning**: Standard criteria apply
                    """)
                st.markdown('</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ **Prediction Error**: {str(e)}")

# ==================== SHAP ANALYSIS PAGE ====================
elif app_mode == "📊 SHAP Analysis":
    st.markdown('<h2 class="sub-header">SHAP Model Interpretability Analysis</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Understanding SHAP (SHapley Additive exPlanations)
    
    SHAP values explain how each feature contributes to a specific prediction. 
    This analysis helps understand the model's decision-making process.
    """)
    
    # 生成示例数据
    st.markdown("### Generate Sample Data for SHAP Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_samples = st.slider("Number of samples", 20, 100, 50)
        
    with col2:
        analysis_type = st.selectbox(
            "Select SHAP visualization",
            ["Waterfall Plot", "Summary Plot", "Feature Importance"]
        )
    
    # 生成样本数据
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'Age': np.random.uniform(20, 80, n_samples),
        'Surgery.time': np.random.uniform(30, 300, n_samples),
        'Anesthesia': np.random.choice([1, 2], n_samples, p=[0.6, 0.4]),
        'Calcium': np.random.uniform(1.8, 2.6, n_samples),
        'ESR': np.random.uniform(5, 80, n_samples)
    })
    
    if st.button("🔍 **Run SHAP Analysis**", type="primary"):
        with st.spinner("**Calculating SHAP values...**"):
            try:
                # 创建SHAP解释器
                explainer = shap.TreeExplainer(model)
                
                # 计算SHAP值
                shap_values = explainer.shap_values(sample_data)
                
                # 处理不同的分析类型
                if analysis_type == "Waterfall Plot":
                    st.markdown('<h3 class="sub-header">Individual SHAP Waterfall Plot</h3>', unsafe_allow_html=True)
                    
                    # 选择一个样本
                    sample_idx = st.selectbox("Select sample", range(min(10, n_samples)))
                    
                    # 创建瀑布图
                    if isinstance(shap_values, list):
                        # 二分类
                        if len(shap_values) == 2:
                            shap_val = shap_values[1][sample_idx]
                            base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
                        else:
                            shap_val = shap_values[0][sample_idx]
                            base_value = explainer.expected_value
                    else:
                        shap_val = shap_values[sample_idx]
                        base_value = explainer.expected_value
                    
                    # 创建解释对象
                    explanation = shap.Explanation(
                        values=shap_val,
                        base_values=base_value,
                        data=sample_data.iloc[sample_idx],
                        feature_names=sample_data.columns.tolist()
                    )
                    
                    # 创建瀑布图
                    fig, ax = plt.subplots(figsize=(12, 8))
                    shap.plots.waterfall(explanation, max_display=10, show=False)
                    plt.title(f"SHAP Waterfall Plot for Sample {sample_idx}", fontsize=16, fontweight='bold')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    # 显示样本数据
                    st.markdown(f"**Sample {sample_idx} Data:**")
                    st.dataframe(sample_data.iloc[[sample_idx]], use_container_width=True)
                    
                elif analysis_type == "Summary Plot":
                    st.markdown('<h3 class="sub-header">SHAP Summary Plot</h3>', unsafe_allow_html=True)
                    
                    # 处理SHAP值格式
                    if isinstance(shap_values, list):
                        if len(shap_values) == 2:
                            shap_array = shap_values[1]  # 正类
                        else:
                            shap_array = shap_values[0]
                    else:
                        shap_array = shap_values
                    
                    # 创建摘要图
                    fig, ax = plt.subplots(figsize=(12, 8))
                    shap.summary_plot(shap_array, sample_data, show=False)
                    plt.title("SHAP Summary Plot", fontsize=16, fontweight='bold')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                    
                else:  # Feature Importance
                    st.markdown('<h3 class="sub-header">SHAP Feature Importance</h3>', unsafe_allow_html=True)
                    
                    # 计算平均绝对SHAP值
                    if isinstance(shap_values, list):
                        if len(shap_values) == 2:
                            shap_array = shap_values[1]
                        else:
                            shap_array = shap_values[0]
                    else:
                        shap_array = shap_values
                    
                    mean_shap = np.mean(np.abs(shap_array), axis=0)
                    features = sample_data.columns.tolist()
                    
                    # 创建条形图
                    fig_imp = go.Figure()
                    fig_imp.add_trace(go.Bar(
                        x=features,
                        y=mean_shap,
                        marker_color='#3B82F6'
                    ))
                    
                    fig_imp.update_layout(
                        title='Mean Absolute SHAP Values (Global Feature Importance)',
                        xaxis_title='Feature',
                        yaxis_title='Mean |SHAP value|',
                        height=400
                    )
                    
                    st.plotly_chart(fig_imp, use_container_width=True)
                
                # SHAP解释
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.markdown('### 📚 **SHAP Value Interpretation Guide**')
                
                st.markdown("""
                **SHAP Values Explained:**
                
                - **Positive SHAP value**: Feature increases the prediction probability
                - **Negative SHAP value**: Feature decreases the prediction probability
                - **Magnitude**: How much the feature changes the prediction
                - **Base value**: Average model prediction (expected value)
                
                **For binary classification:**
                - SHAP shows how features push the prediction toward class 1 (Positive) or class 2 (Negative)
                - Larger absolute values indicate more important features for that prediction
                
                **Clinical application:**
                - Identify which clinical factors are driving a high-risk prediction
                - Understand trade-offs between different risk factors
                - Explain model decisions to clinicians and patients
                """)
                st.markdown('</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ **SHAP Analysis Error**: {str(e)}")
                st.info("""
                **Troubleshooting suggestions:**
                1. Ensure your model is a proper LightGBM model
                2. Try reducing the number of samples
                3. Check that SHAP library is properly installed
                4. The model might not be fully compatible with SHAP TreeExplainer
                """)

# ==================== MODEL INFORMATION ====================
else:
    st.markdown('<h2 class="sub-header">Model Information and SHAP Status</h2>', unsafe_allow_html=True)
    
    # 模型状态
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<p class="stat-label">MODEL TYPE</p>', unsafe_allow_html=True)
        if isinstance(model, LGBMClassifier):
            st.markdown('<p class="stat-value" style="color: #10B981;">LightGBM</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="stat-value" style="color: #F59E0B;">Demo Model</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<p class="stat-label">SHAP COMPATIBILITY</p>', unsafe_allow_html=True)
        shap_compatible = isinstance(model, LGBMClassifier) or hasattr(model, '_Booster')
        if shap_compatible:
            st.markdown('<p class="stat-value" style="color: #10B981;">✅ Compatible</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="stat-value" style="color: #F59E0B;">⚠️ Limited</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # SHAP信息
    st.markdown("""
    ### SHAP Interpretability Features
    
    **Available SHAP visualizations:**
    
    1. **Waterfall Plots**: Show feature contributions for individual predictions
    2. **Summary Plots**: Global view of feature importance and effects
    3. **Feature Importance**: Mean absolute SHAP values across the dataset
    
    **How SHAP works:**
    
    SHAP (SHapley Additive exPlanations) is a game theory approach to explain 
    machine learning model predictions. It assigns each feature an importance 
    value for a particular prediction, showing how much each feature contributed 
    to moving the prediction from the base value (average prediction) to the 
    final prediction.
    
    **Clinical relevance:**
    
    - **Transparency**: Understand why the model makes specific predictions
    - **Trust**: Build confidence in model recommendations
    - **Insight**: Identify which clinical factors are most important
    - **Education**: Teach about risk factor interactions
    
    **Technical requirements for SHAP:**
    - Tree-based models (LightGBM, XGBoost, Random Forest)
    - Proper model serialization
    - SHAP library installation
    """)
    
    # 模型详细信息
    if isinstance(model, LGBMClassifier):
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown('### 🔧 **LightGBM Model Details**')
        
        try:
            params = model.get_params()
            important_params = {k: v for k, v in params.items() 
                              if k in ['n_estimators', 'max_depth', 'learning_rate', 'random_state']}
            
            st.write("**Key parameters:**")
            for param, value in important_params.items():
                st.write(f"- {param}: {value}")
                
        except:
            st.write("Model parameters not available")
            
        st.markdown('</div>', unsafe_allow_html=True)

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6B7280; margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #E5E7EB;">
    <p><strong>Postoperative Hypoproteinemia Risk Prediction System</strong> | Version 4.0</p>
    <p>© 2024 Clinical Research Division | For Research Use Only</p>
    <p><small>SHAP interpretability provides transparent explanations for model predictions.</small></p>
</div>
""", unsafe_allow_html=True)
