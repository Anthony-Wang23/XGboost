import pandas as pd
import shap
from xgboost import XGBClassifier

import streamlit as st

# 加载 XGBoost 模型
model_path = 'D:/anaconda3/envs/py312/xgboost_model.bin'  # XGBoost 模型的路径

# 加载模型
model = XGBClassifier()
model.load_model(model_path)  # 加载模型

# 设置 Streamlit 应用的标题
st.title("黑色素瘤前哨淋巴结转移概率预测")

# 在侧边栏中输入特征
st.sidebar.header("输入特征")
Breslow_Thickness = st.sidebar.slider("Breslow_厚度(mm)", min_value=0.0, value=4.0, step=0.1)
Ki67 = st.sidebar.slider("Ki67 (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
Subtype = st.sidebar.radio("是否为甲下型？", options=[0, 1], format_func=lambda x: "否" if x == 0 else "是")
Supplementary_Check = st.sidebar.radio("是否已经进行治疗？", options=[0, 1], format_func=lambda x: "无" if x == 0 else "有")

# 准备输入数据
input_data = pd.DataFrame({
    'Breslow_Thickness': [Breslow_Thickness],
    'Ki67': [Ki67],
    'Subtype': [Subtype],
    'Supplementary_Check': [Supplementary_Check]
})



# 添加预测按钮，用户点击后进行模型预测
if st.button("预测"):
    # 使用 XGBoost 模型进行预测，输出概率
    # 获取预测的概率（确保使用 predict_proba 而非 predict）
    predicted_probs = model.predict_proba(input_data)

    # 如果是二维数组，提取类别 1 的概率
    probabilities = predicted_probs[:, 0]

    # 设置自定义阈值
    threshold = 0.49944069981575

    # 根据阈值判断概率并输出相应的信息
    if probabilities >= threshold:
        message = '患者黑色素瘤前哨淋巴结转移的概率较大'
    else:
        message = '患者黑色素瘤前哨淋巴结转移的概率较小'



    st.markdown(f"<h3 style='font-size: 36px; font-weight: bold;'>{message}</h3>", unsafe_allow_html=True)

    # 初始化 SHAP 的 JavaScript 文件，确保交互性
    shap.initjs()

    # 创建 SHAP 解释器对象
    explainer = shap.Explainer(model)

    # 使用解释器计算输入数据的 SHAP 值
    shap_values = explainer(input_data)

    # 检查 SHAP 值是否计算成功
    if len(shap_values) > 0:
        sample_shap_values = shap_values[0].values
        expected_value = explainer.expected_value

        # 事件1的期望值（反转事件0的期望值）
        expected_value_event1 = 1 - expected_value

        # 反转 SHAP 值符号，以适应事件1的预测
        sample_shap_values_event1 = -sample_shap_values
    else:
        st.error("SHAP 值计算出错！")
        sample_shap_values_event1 = None

    # 如果 SHAP 值计算成功，创建解释对象并展示力图
    if sample_shap_values_event1 is not None:
        explanation = shap.Explanation(
            values=sample_shap_values_event1,
            base_values=expected_value_event1,
            data=input_data.iloc[0].values,
            feature_names=input_data.columns.tolist()
        )

        # 保存 SHAP 力图
        shap.save_html("hap_force_plot.html", shap.plots.force(explanation, show=False))

        # 显示 SHAP 力图和解释文字
        st.subheader("模型预测的力图")
        with open("shap_force_plot.html", encoding='utf-8') as f:
            st.components.v1.html(f.read(), height=200)

        # 添加 SHAP 力图的解释
        st.markdown("""
        **SHAP 力图解释：**

        SHAP 力图展示了每个特征对黑色素瘤前哨淋巴结转移预测的贡献。每个特征值的条形表示了该特征对最终预测结果的影响力。具体来说：

        - 每个特征值的大小和颜色代表了该特征对预测概率的推动作用：红色表示该特征值增加了预测为“转移”的概率，而蓝色则表示该特征值减少了这种概率。
        - 长条越长，说明该特征对模型预测的影响越大。
        - 基准线代表了模型的“期望值”，即如果没有任何特征信息时，模型预测的概率。

        通过 SHAP 力图，你可以清楚地看到哪些特征对模型的预测影响最大，哪些特征推动了预测结果朝向“转移”或“未转移”方向。
        """)

#streamlit run D:/anaconda3/envs/py312/streamlit.py运行文件