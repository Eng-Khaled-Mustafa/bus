import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ---- Fake model training to simulate logic ----
def train_model():
    np.random.seed(42)
    data = pd.DataFrame({
        'EngineTemp': np.random.normal(90, 10, 10000),
        'OilPressure': np.random.normal(3.5, 0.5, 10000),
        'RPM': np.random.randint(600, 2500, 10000),
        'ErrorCode': np.random.choice(['None', 'P0300', 'P0420', 'P0171', 'P0401'], 10000, p=[0.9, 0.03, 0.03, 0.02, 0.02]),
        'KM_Today': np.random.randint(100, 300, 10000),
    })

    data['WillNeedMaintenance'] = (
        (data['EngineTemp'] > 100).astype(int)
        + (data['OilPressure'] < 2.5).astype(int)
        + (data['RPM'] > 2200).astype(int)
        + (data['ErrorCode'] != 'None').astype(int)
    ) >= 2

    le = LabelEncoder()
    data['ErrorCode'] = le.fit_transform(data['ErrorCode'])

    X = data[['EngineTemp', 'OilPressure', 'RPM', 'ErrorCode', 'KM_Today']]
    y = data['WillNeedMaintenance']

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    return model, le

model, le = train_model()

st.title("محاكاة ذكية لصيانة الباصات")

st.markdown("""
🚌 أدخل بيانات تشغيل الباص أدناه، وسنخبرك:
- هل يحتاج إلى صيانة؟
- كم نسبة الاحتمال؟
- هل يدخل جدول الصيانة؟
""")

# ---- إدخال البيانات ----
with st.form("bus_form"):
    engine_temp = st.number_input("درجة حرارة المحرك", value=90.0)
    oil_pressure = st.number_input("ضغط الزيت (bar)", value=3.5)
    rpm = st.number_input("سرعة الدوران RPM", value=1000)
    error_code = st.selectbox("رمز الخطأ (DTC)", ['None', 'P0300', 'P0420', 'P0171', 'P0401'])
    km_today = st.number_input("المسافة المقطوعة اليوم (كم)", value=200)
    submitted = st.form_submit_button("تحليل الحالة")

if submitted:
    error_encoded = le.transform([error_code])[0]
    input_data = pd.DataFrame([[engine_temp, oil_pressure, rpm, error_encoded, km_today]],
                              columns=['EngineTemp', 'OilPressure', 'RPM', 'ErrorCode', 'KM_Today'])

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    st.subheader("🔎 النتيجة")
    st.write(f"📊 احتمالية الحاجة إلى الصيانة: **{prob:.2%}**")

    if prob > 0.7:
        st.error("🚨 هذا الباص مرشح لدخول جدول الصيانة!")
    else:
        st.success("✅ لا حاجة فورية للصيانة")
