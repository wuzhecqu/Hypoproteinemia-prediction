import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import shap
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from lightgbm import LGBMClassifier

# 设置页面配置
st.set_page_config(
    page_title="术后低蛋白血症预测系统",
    page_icon="🏥",
    layout="wide"
)

# 应用标题
st.title("🏥 术后低蛋白血症风险预测系统")
st.markdown("---")

# 缓存加载模型
@st.cache_resource
def load_model():
    try:
        # 尝试从pickle文件加载模型
        with open('lgb_model_weights.pkl', 'rb') as f:
            model = pickle.load(f)
        st.success("✅ 模型加载成功!")
        return model
    except Exception as e:
        st.error(f"❌ 模型加载失败: {e}")
        return None

# 加载模型
model = load_model()

# 如果模型加载成功，显示模型信息
if model:
    st.sidebar.success(f"模型已加载 (LightGBM)")
    
    # 显示特征重要性（如果有）
    if hasattr(model, 'feature_importances_'):
        st.sidebar.info("特征已准备")
    
    # 特征描述
    feature_descriptions = {
        'Age': '患者年龄（岁）',
        'Surgery.time': '手术时长（分钟）',
        'Anesthesia': '麻醉类型（1: 全身麻醉, 2: 椎管内麻醉, 3: 局部麻醉）',
        'Calcium': '血清钙水平（mmol/L）',
        'ESR': '红细胞沉降率（mm/h）'
    }

# 创建标签映射
label_map = {1: "有低蛋白血症", 2: "无低蛋白血症"}

# 创建标签反向映射
reverse_label_map = {"有低蛋白血症": 1, "无低蛋白血症": 2}

# 创建标签映射用于SHAP解释
label_map_shap = {1: 1, 2: 0}  # 1: 有低蛋白血症, 0: 无低蛋白血症

# 侧边栏 - 导航
st.sidebar.title("🔍 导航")
app_mode = st.sidebar.selectbox(
    "请选择功能",
    ["📊 单样本预测", "📈 SHAP可解释性分析", "📋 验证集批量预测", "📝 使用说明"]
)

# 功能1: 单样本预测
if app_mode == "📊 单样本预测":
    st.header("单样本预测")
    st.markdown("请输入患者的临床参数进行预测")
    
    # 创建两列布局
    col1, col2 = st.columns(2)
    
    with col1:
        Age = st.number_input(
            "年龄（岁）", 
            min_value=0, 
            max_value=120, 
            value=50,
            help="患者年龄"
        )
        
        Surgery_time = st.number_input(
            "手术时长（分钟）", 
            min_value=0, 
            max_value=600, 
            value=120,
            help="手术持续时间"
        )
        
        Anesthesia = st.selectbox(
            "麻醉类型",
            ["全身麻醉", "椎管内麻醉", "局部麻醉"],
            help="选择麻醉方式"
        )
    
    with col2:
        Calcium = st.number_input(
            "血清钙（mmol/L）", 
            min_value=1.0, 
            max_value=3.5, 
            value=2.2,
            step=0.1,
            help="血清钙水平"
        )
        
        ESR = st.number_input(
            "红细胞沉降率（mm/h）", 
            min_value=0, 
            max_value=150, 
            value=20,
            help="ESR值"
        )
    
    # 转换麻醉类型为数值
    anesthesia_map = {"全身麻醉": 1, "椎管内麻醉": 2, "局部麻醉": 3}
    Anesthesia_numeric = anesthesia_map[Anesthesia]
    
    # 创建输入数据框
    input_data = pd.DataFrame({
        'Age': [Age],
        'Surgery.time': [Surgery_time],
        'Anesthesia': [Anesthesia_numeric],
        'Calcium': [Calcium],
        'ESR': [ESR]
    })
    
    # 预测按钮
    if st.button("🔮 开始预测", type="primary"):
        if model:
            try:
                # 进行预测
                prediction = model.predict(input_data)[0]
                prediction_proba = model.predict_proba(input_data)[0]
                
                # 获取预测概率
                prob_class1 = prediction_proba[0]  # 有低蛋白血症的概率
                prob_class2 = prediction_proba[1]  # 无低蛋白血症的概率
                
                # 显示结果
                st.markdown("---")
                st.subheader("📋 预测结果")
                
                # 创建结果卡片
                result_col1, result_col2 = st.columns(2)
                
                with result_col1:
                    st.metric(
                        label="预测类别",
                        value=label_map[prediction],
                        delta=f"置信度: {max(prob_class1, prob_class2)*100:.1f}%"
                    )
                
                with result_col2:
                    risk_color = "🟢" if prediction == 2 else "🔴"
                    st.metric(
                        label="风险评估",
                        value=f"{risk_color} {'低风险' if prediction == 2 else '高风险'}"
                    )
                
                # 显示概率分布
                st.subheader("📊 概率分布")
                
                # 创建概率条形图
                fig_prob = go.Figure()
                
                fig_prob.add_trace(go.Bar(
                    x=['有低蛋白血症', '无低蛋白血症'],
                    y=[prob_class1, prob_class2],
                    text=[f'{prob_class1*100:.1f}%', f'{prob_class2*100:.1f}%'],
                    textposition='auto',
                    marker_color=['#EF553B', '#00CC96']
                ))
                
                fig_prob.update_layout(
                    title='预测概率分布',
                    xaxis_title='类别',
                    yaxis_title='概率',
                    yaxis=dict(range=[0, 1]),
                    height=400
                )
                
                st.plotly_chart(fig_prob, use_container_width=True)
                
                # 显示输入值
                st.subheader("📝 输入参数")
                input_display = input_data.copy()
                input_display['麻醉类型'] = Anesthesia
                st.dataframe(input_display.drop('Anesthesia', axis=1), use_container_width=True)
                
            except Exception as e:
                st.error(f"预测过程中出现错误: {e}")

# 功能2: SHAP可解释性分析
elif app_mode == "📈 SHAP可解释性分析":
    st.header("SHAP可解释性分析")
    st.markdown("此功能用于解释模型预测结果")
    
    if model:
        # 创建示例数据或使用用户输入
        st.info("🔍 请先使用单样本预测功能生成预测，然后分析可解释性")
        
        # 获取特征名称
        feature_names = ['Age', 'Surgery.time', 'Anesthesia', 'Calcium', 'ESR']
        
        # 创建示例数据
        example_data = pd.DataFrame({
            'Age': [60],
            'Surgery.time': [180],
            'Anesthesia': [1],
            'Calcium': [2.0],
            'ESR': [35]
        })
        
        # 计算SHAP值
        try:
            # 创建SHAP解释器
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(example_data)
            
            # 显示SHAP摘要图
            st.subheader("📊 SHAP特征重要性")
            
            # 创建SHAP值的条形图
            if isinstance(shap_values, list):
                # 对于分类问题，取第一个类别的SHAP值
                shap_array = shap_values[0][0]
            else:
                shap_array = shap_values[0]
            
            # 创建特征重要性数据框
            shap_df = pd.DataFrame({
                '特征': feature_names,
                'SHAP值': np.abs(shap_array),
                '方向': ['正向' if x > 0 else '负向' for x in shap_array]
            }).sort_values('SHAP值', ascending=True)
            
            # 创建水平条形图
            fig_shap = go.Figure()
            
            colors = ['#00CC96' if dir == '正向' else '#EF553B' for dir in shap_df['方向']]
            
            fig_shap.add_trace(go.Bar(
                y=shap_df['特征'],
                x=shap_df['SHAP值'],
                orientation='h',
                marker_color=colors,
                text=[f'{val:.3f}' for val in shap_df['SHAP值']],
                textposition='auto'
            ))
            
            fig_shap.update_layout(
                title='特征对预测结果的影响程度',
                xaxis_title='SHAP值（绝对值）',
                yaxis_title='特征',
                height=400
            )
            
            st.plotly_chart(fig_shap, use_container_width=True)
            
            # 解释说明
            st.subheader("📝 解释说明")
            st.markdown("""
            **SHAP值解释：**
            - **正值（绿色）**：增加该特征值会提高"有低蛋白血症"的风险
            - **负值（红色）**：增加该特征值会降低"有低蛋白血症"的风险
            
            **特征说明：**
            1. **ESR**：红细胞沉降率，数值越高通常表示炎症反应越强
            2. **年龄**：年龄越大，术后并发症风险可能越高
            3. **手术时长**：手术时间越长，身体应激反应可能越强
            4. **血清钙**：钙离子参与多种生理过程，异常值可能影响恢复
            5. **麻醉类型**：不同麻醉方式对生理影响不同
            """)
            
        except Exception as e:
            st.warning(f"SHAP分析遇到问题: {e}")
            st.info("这可能是由于SHAP版本兼容性问题。您仍然可以使用模型进行预测。")

# 功能3: 验证集批量预测
elif app_mode == "📋 验证集批量预测":
    st.header("验证集批量预测")
    
    # 上传验证集文件
    uploaded_file = st.file_uploader(
        "上传验证集Excel文件", 
        type=['xlsx', 'xls'],
        help="请上传包含以下列的Excel文件：Age, Surgery.time, Anesthesia, Calcium, ESR"
    )
    
    if uploaded_file is not None:
        try:
            # 读取Excel文件
            validation_data = pd.read_excel(uploaded_file)
            
            # 检查必要的列
            required_columns = ['Age', 'Surgery.time', 'Anesthesia', 'Calcium', 'ESR']
            missing_columns = [col for col in required_columns if col not in validation_data.columns]
            
            if missing_columns:
                st.error(f"文件缺少以下必要列: {missing_columns}")
            else:
                # 显示数据预览
                st.subheader("📊 数据预览")
                st.dataframe(validation_data.head(10), use_container_width=True)
                st.info(f"数据形状: {validation_data.shape[0]} 行 × {validation_data.shape[1]} 列")
                
                # 预测按钮
                if st.button("🔮 批量预测", type="primary"):
                    if model:
                        with st.spinner("正在进行批量预测..."):
                            # 进行预测
                            predictions = model.predict(validation_data[required_columns])
                            prediction_probas = model.predict_proba(validation_data[required_columns])
                            
                            # 添加预测结果到数据框
                            results_df = validation_data.copy()
                            results_df['预测结果'] = [label_map[p] for p in predictions]
                            results_df['有低蛋白血症概率'] = prediction_probas[:, 0]
                            results_df['无低蛋白血症概率'] = prediction_probas[:, 1]
                            
                            # 计算准确率（如果有真实标签）
                            if 'Hypoproteinemia' in results_df.columns:
                                results_df['真实结果'] = [label_map.get(int(x), f"未知({x})") 
                                                        if pd.notna(x) else "未知" 
                                                        for x in results_df['Hypoproteinemia']]
                                results_df['预测正确'] = results_df['预测结果'] == results_df['真实结果']
                                accuracy = results_df['预测正确'].mean() * 100
                                
                                st.success(f"✅ 批量预测完成！准确率: {accuracy:.2f}%")
                            else:
                                st.success("✅ 批量预测完成！")
                            
                            # 显示预测结果
                            st.subheader("📋 预测结果")
                            st.dataframe(results_df, use_container_width=True)
                            
                            # 统计预测分布
                            st.subheader("📈 预测结果分布")
                            
                            prediction_counts = results_df['预测结果'].value_counts()
                            fig_dist = go.Figure(data=[
                                go.Pie(
                                    labels=prediction_counts.index,
                                    values=prediction_counts.values,
                                    hole=.3
                                )
                            ])
                            
                            fig_dist.update_layout(
                                title='预测结果分布'
                            )
                            
                            st.plotly_chart(fig_dist, use_container_width=True)
                            
                            # 提供下载链接
                            csv = results_df.to_csv(index=False).encode('utf-8-sig')
                            st.download_button(
                                label="📥 下载预测结果 (CSV)",
                                data=csv,
                                file_name="batch_prediction_results.csv",
                                mime="text/csv"
                            )
        
        except Exception as e:
            st.error(f"读取文件时出错: {e}")

# 功能4: 使用说明
else:
    st.header("📝 使用说明")
    
    st.markdown("""
    ## 术后低蛋白血症预测系统使用指南
    
    ### 📊 单样本预测
    1. 在左侧导航栏选择"📊 单样本预测"
    2. 输入患者的临床参数：
       - **年龄**：患者年龄（岁）
       - **手术时长**：手术持续时间（分钟）
       - **麻醉类型**：选择麻醉方式
       - **血清钙**：血清钙水平（mmol/L）
       - **ESR**：红细胞沉降率（mm/h）
    3. 点击"🔮 开始预测"按钮
    4. 查看预测结果和概率分布
    
    ### 📈 SHAP可解释性分析
    1. 在左侧导航栏选择"📈 SHAP可解释性分析"
    2. 系统将展示特征对预测结果的影响程度
    3. 了解哪些因素对预测结果贡献最大
    
    ### 📋 验证集批量预测
    1. 在左侧导航栏选择"📋 验证集批量预测"
    2. 上传包含患者数据的Excel文件
    3. 文件应包含以下列：Age, Surgery.time, Anesthesia, Calcium, ESR
    4. 点击"🔮 批量预测"按钮
    5. 查看和下载预测结果
    
    ### 📁 文件要求
    - 模型文件：`lgb_model_weights.pkl`
    - 验证集文件：Excel格式，包含必要的临床参数
    
    ### ⚠️ 注意事项
    - 确保输入数据在合理范围内
    - 模型预测结果仅供参考，实际临床决策需结合专业知识
    - 如遇问题，请检查文件格式和数据完整性
    """)

# 页面底部信息
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    <p>术后低蛋白血症预测系统 v1.0 | 仅供临床研究参考使用</p>
    </div>
    """,
    unsafe_allow_html=True
)
