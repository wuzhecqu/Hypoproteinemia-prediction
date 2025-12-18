# -*- coding: utf-8 -*-
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# ====================== 配置 ======================
st.set_page_config(page_title="Hypoproteinemia Prediction", layout="wide")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 英语界面，无需中文字体
plt.rcParams['axes.unicode_minus'] = False


# ====================== 加载模型权重 ======================
@st.cache_resource  # 缓存模型，避免重复加载
def load_model():
    try:
        with open("lgb_model_weights.pkl", "rb") as f:
            model_metadata = pickle.load(f)
        return model_metadata
    except FileNotFoundError:
        st.error("Model weights file 'lgb_model_weights.pkl' not found!")
        st.stop()


model_metadata = load_model()
model = model_metadata['model']
imputer = model_metadata['imputer']
scaler = model_metadata['scaler']
feature_cols = model_metadata['feature_cols']
target_mapping = model_metadata['target_mapping']
feature_descs = model_metadata['feature_descriptions']

# ====================== 网页界面（英语） ======================
st.title("Postoperative Hypoproteinemia Prediction (LightGBM)")
st.subheader("Clinical Decision Support System")

# 分割页面：左侧输入，右侧结果
col1, col2 = st.columns([1, 2])

# ---------------------- 左侧：单样本输入 ----------------------
with col1:
    st.header("1. Single Sample Input")
    st.write("Enter patient features to predict hypoproteinemia risk:")

    # 生成特征输入框（数值型）
    input_features = {}
    for feat in feature_cols:
        input_features[feat] = st.number_input(
            label=f"{feat}: {feature_descs[feat]}",
            value=0.0,
            step=0.1,
            format="%.1f"
        )

    # 预测按钮
    predict_btn = st.button("Predict Risk", type="primary")

# ---------------------- 右侧：预测结果 + 可解释性 ----------------------
with col2:
    if predict_btn:
        # 步骤1：整理输入数据
        input_df = pd.DataFrame([input_features])
        st.subheader("Input Features")
        st.dataframe(input_df, use_container_width=True)

        # 步骤2：预处理（缺失值填充 + 标准化）
        input_imputed = imputer.transform(input_df)
        input_scaled = scaler.transform(input_imputed)

        # 步骤3：模型预测
        pred_prob = model.predict_proba(input_scaled)[0, 1]  # 阳性概率
        pred_label = 1 if pred_prob >= 0.5 else 0
        pred_class = target_mapping[pred_label]

        # 显示预测结果
        st.subheader("2. Prediction Result")
        col_result1, col_result2 = st.columns(2)
        with col_result1:
            st.metric(
                label="Hypoproteinemia Risk Probability",
                value=f"{pred_prob:.2%}",
                delta=f"Threshold: 50%"
            )
        with col_result2:
            st.metric(
                label="Prediction Class",
                value=pred_class,
                delta="High Risk" if pred_prob >= 0.5 else "Low Risk"
            )

        # 步骤4：可解释性分析（SHAP）
        st.subheader("3. Model Interpretability (SHAP Analysis)")

        # 初始化SHAP解释器（LightGBM用TreeExplainer）
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_scaled)

        # 子图1：Force Plot（单样本解释）
        st.write("### SHAP Force Plot (Single Sample Explanation)")
        shap.initjs()  # 启用JS渲染
        force_plot = shap.force_plot(
            explainer.expected_value,
            shap_values[0],
            input_df.iloc[0],
            feature_names=feature_cols,
            matplotlib=False,
            show=False
        )
        st.components.v1.html(shap.getjs() + force_plot.html(), height=150)

        # 子图2：Summary Plot（特征重要性，用验证集数据增强解释性）
        st.write("### SHAP Summary Plot (Feature Importance)")
        # 加载验证集数据生成Summary Plot（可选，增强解释性）
        val_data = pd.read_excel("validation_data.xlsx")
        val_data.columns = [col.strip() for col in val_data.columns]
        val_X = val_data[feature_cols]
        val_X_imputed = imputer.transform(val_X)
        val_X_scaled = scaler.transform(val_X_imputed)
        val_shap_values = explainer.shap_values(val_X_scaled)

        # 绘制Summary Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(
            val_shap_values,
            val_X,
            feature_names=feature_cols,
            plot_type="dot",
            show=False,
            ax=ax
        )
        st.pyplot(fig, use_container_width=True)

        # 子图3：Feature Dependence Plot（Top 1 Feature）
        st.write("### SHAP Dependence Plot (Top 1 Feature)")
        # 计算特征重要性
        feature_importance = np.abs(val_shap_values).mean(axis=0)
        top_feat_idx = np.argmax(feature_importance)
        top_feat = feature_cols[top_feat_idx]

        fig2, ax2 = plt.subplots(figsize=(8, 5))
        shap.dependence_plot(
            top_feat,
            val_shap_values,
            val_X,
            feature_names=feature_cols,
            show=False,
            ax=ax2
        )
        st.pyplot(fig2, use_container_width=True)

# ---------------------- 页脚 ----------------------
st.markdown("---")
st.caption("Disclaimer: This tool is for research use only, not for clinical diagnosis.")