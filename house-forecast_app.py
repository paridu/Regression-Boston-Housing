# Boston Housing
#Data set :https://www.kaggle.com/c/boston-housing/data

import streamlit as st  # นำเข้า Streamlit เพื่อสร้างแอปพลิเคชันเว็บ
import pandas as pd  # นำเข้า Pandas เพื่อจัดการข้อมูลในรูป DataFrame
import shap  # นำเข้า SHAP เพื่ออธิบายการทำนายของโมเดล
import matplotlib.pyplot as plt  # นำเข้า Matplotlib เพื่อสร้างกราฟ
from sklearn import datasets  # นำเข้า datasets จาก Scikit-learn เพื่อโหลดข้อมูลตัวอย่าง
from sklearn.ensemble import RandomForestRegressor  # นำเข้า RandomForestRegressor เพื่อสร้างโมเดล

st.write("""
# Boston House Price Prediction App

This app predicts the **Boston House Price**!
""")  # แสดงหัวข้อและคำอธิบายของแอปพลิเคชัน
st.write('---')  # แสดงเส้นแบ่ง

# Loads the Boston House Price Dataset
boston = datasets.load_boston()  # โหลดข้อมูล Boston House Price
X = pd.DataFrame(boston.data, columns=boston.feature_names)  # สร้าง DataFrame สำหรับข้อมูล features
Y = pd.DataFrame(boston.target, columns=["MEDV"])  # สร้าง DataFrame สำหรับข้อมูล target

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')  # แสดงหัวข้อของ Sidebar

def user_input_features():
    CRIM = st.sidebar.slider('CRIM', X.CRIM.min(), X.CRIM.max(), X.CRIM.mean())  # สร้าง Slider สำหรับค่า CRIM
    ZN = st.sidebar.slider('ZN', X.ZN.min(), X.ZN.max(), X.ZN.mean())  # สร้าง Slider สำหรับค่า ZN
    INDUS = st.sidebar.slider('INDUS', X.INDUS.min(), X.INDUS.max(), X.INDUS.mean())  # สร้าง Slider สำหรับค่า INDUS
    CHAS = st.sidebar.slider('CHAS', X.CHAS.min(), X.CHAS.max(), X.CHAS.mean())  # สร้าง Slider สำหรับค่า CHAS
    NOX = st.sidebar.slider('NOX', X.NOX.min(), X.NOX.max(), X.NOX.mean())  # สร้าง Slider สำหรับค่า NOX
    RM = st.sidebar.slider('RM', X.RM.min(), X.RM.max(), X.RM.mean())  # สร้าง Slider สำหรับค่า RM
    AGE = st.sidebar.slider('AGE', X.AGE.min(), X.AGE.max(), X.AGE.mean())  # สร้าง Slider สำหรับค่า AGE
    DIS = st.sidebar.slider('DIS', X.DIS.min(), X.DIS.max(), X.DIS.mean())  # สร้าง Slider สำหรับค่า DIS
    RAD = st.sidebar.slider('RAD', X.RAD.min(), X.RAD.max(), X.RAD.mean())  # สร้าง Slider สำหรับค่า RAD
    TAX = st.sidebar.slider('TAX', X.TAX.min(), X.TAX.max(), X.TAX.mean())  # สร้าง Slider สำหรับค่า TAX
    PTRATIO = st.sidebar.slider('PTRATIO', X.PTRATIO.min(), X.PTRATIO.max(), X.PTRATIO.mean())  # สร้าง Slider สำหรับค่า PTRATIO
    B = st.sidebar.slider('B', X.B.min(), X.B.max(), X.B.mean())  # สร้าง Slider สำหรับค่า B
    LSTAT = st.sidebar.slider('LSTAT', X.LSTAT.min(), X.LSTAT.max(), X.LSTAT.mean())  # สร้าง Slider สำหรับค่า LSTAT
    data = {'CRIM': CRIM,
            'ZN': ZN,
            'INDUS': INDUS,
            'CHAS': CHAS,
            'NOX': NOX,
            'RM': RM,
            'AGE': AGE,
            'DIS': DIS,
            'RAD': RAD,
            'TAX': TAX,
            'PTRATIO': PTRATIO,
            'B': B,
            'LSTAT': LSTAT}  # สร้าง Dictionary สำหรับเก็บค่าจาก Slider
    features = pd.DataFrame(data, index=[0])  # สร้าง DataFrame จาก Dictionary
    return features  # คืนค่า DataFrame

df = user_input_features()  # เรียกใช้ฟังก์ชันเพื่อรับค่าจากผู้ใช้

# Main Panel

# Print specified input parameters
st.header('Specified Input parameters')  # แสดงหัวข้อของการแสดงค่าที่ผู้ใช้เลือก
st.write(df)  # แสดงค่าที่ผู้ใช้เลือกในรูป DataFrame
st.write('---')  # แสดงเส้นแบ่ง

# Build Regression Model
model = RandomForestRegressor()  # สร้างโมเดล RandomForestRegressor
model.fit(X, Y)  # ฝึกฝนโมเดลด้วยข้อมูล X และ Y
# Apply Model to Make Prediction
prediction = model.predict(df)  # ทำนายค่าจากโมเดลด้วยข้อมูลที่ผู้ใช้ป้อน

st.header('Prediction of MEDV')  # แสดงหัวข้อของการทำนาย
st.write(prediction)  # แสดงค่าทำนาย
st.write('---')  # แสดงเส้นแบ่ง

# Explaining the model's predictions using SHAP values
# https://github.com/slundberg/shap
explainer = shap.TreeExplainer(model)  # สร้างตัวอธิบายของ SHAP สำหรับโมเดล
shap_values = explainer.shap_values(X)  # คำนวณค่า SHAP values

st.header('Feature Importance')  # แสดงหัวข้อของ Feature Importance
plt.title('Feature importance based on SHAP values')  # ตั้งชื่อกราฟ
shap.summary_plot(shap_values, X)  # สร้างกราฟสรุปของ SHAP values
st.pyplot(bbox_inches='tight')  # แสดงกราฟใน Streamlit
st.write('---')  # แสดงเส้นแบ่ง

plt.title('Feature importance based on SHAP values (Bar)')  # ตั้งชื่อกราฟแบบ Bar
shap.summary_plot(shap_values, X, plot_type="bar")  # สร้างกราฟแบบ Bar ของ SHAP values
st.pyplot(bbox_inches='tight')  # แสดงกราฟใน Streamlit
