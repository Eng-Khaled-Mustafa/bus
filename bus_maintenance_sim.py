import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(layout="wide")

# ---- Fake training with historical data ----
def train_model():
    np.random.seed(42)
    data = pd.DataFrame({
        'EngineTemp': np.random.normal(90, 10, 5000),
        'OilPressure': np.random.normal(3.5, 0.5, 5000),
        'RPM': np.random.randint(600, 2500, 5000),
        'ErrorCode': np.random.choice(['None', 'P0300', 'P0420', 'P0171', 'P0401'], 5000, p=[0.85, 0.05, 0.04, 0.03, 0.03]),
        'KM_Today': np.random.randint(100, 300, 5000),
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

st.title("🚐 Smart Garage Maintenance Simulation")
st.markdown("""
🔧 **Interactive Simulation for Garage Prioritization**
- Simulates priority queue for 20 buses
- Tracks how many times a bus enters garage based on threshold (0.7)
- Live chart updates when data changes
- Includes Gantt chart of daily garage schedule
""")

# ---- Generate synthetic bus data ----
n_buses = 20
dates = pd.date_range("2025-07-01", periods=30)
bus_ids = [f"BUS_{i+1:03}" for i in range(n_buses)]

records = []
for bus in bus_ids:
    for date in dates:
        rec = {
            'BusID': bus,
            'Date': date,
            'EngineTemp': np.random.normal(90, 10),
            'OilPressure': np.random.normal(3.5, 0.5),
            'RPM': np.random.randint(600, 2500),
            'ErrorCode': np.random.choice(['None', 'P0300', 'P0420', 'P0171', 'P0401'], p=[0.85, 0.05, 0.04, 0.03, 0.03]),
            'KM_Today': np.random.randint(100, 300),
        }
        records.append(rec)

full_data = pd.DataFrame(records)
full_data['ErrorCode'] = le.transform(full_data['ErrorCode'])
X = full_data[['EngineTemp', 'OilPressure', 'RPM', 'ErrorCode', 'KM_Today']]
full_data['Predicted'] = model.predict_proba(X)[:, 1]
full_data['Scheduled'] = full_data['Predicted'] > 0.7

# جدولة لا تتعدى 10 باصات يوميًا
scheduled_data = full_data[full_data['Scheduled']].copy()
scheduled_data = scheduled_data.sort_values(by=['Date', 'Predicted'], ascending=[True, False])
scheduled_data['DailyCount'] = scheduled_data.groupby('Date').cumcount() + 1
scheduled_data = scheduled_data[scheduled_data['DailyCount'] <= 10]

# Count entries to garage
garage_counts = scheduled_data.groupby('BusID').size().reset_index(name='GarageEntries')

# تحديث full_data من النسخة المحدثة للعرض لاحقًا
full_data = updated_data.copy().size().reset_index(name='GarageEntries')

# ---- Interactive controls ----
st.sidebar.header("🔧 Modify Bus Parameters")
selected_bus = st.sidebar.selectbox("Select BusID", bus_ids)

default_row = full_data[full_data['BusID'] == selected_bus].iloc[-1]

error_options = ['None', 'P0300', 'P0420', 'P0171', 'P0401']
current_error_label = le.inverse_transform([int(default_row['ErrorCode'])])[0]

temp = st.sidebar.slider("Engine Temp", 60.0, 120.0, float(default_row['EngineTemp']))
oil = st.sidebar.slider("Oil Pressure", 1.0, 5.0, float(default_row['OilPressure']))
rpm = st.sidebar.slider("RPM", 600, 2500, int(default_row['RPM']))
error = st.sidebar.selectbox("Error Code", error_options, index=error_options.index(current_error_label))
km = st.sidebar.slider("KM Today", 50, 400, int(default_row['KM_Today']))

encoded_error = le.transform([error])[0]
new_data = pd.DataFrame([[temp, oil, rpm, encoded_error, km]], columns=['EngineTemp', 'OilPressure', 'RPM', 'ErrorCode', 'KM_Today'])
new_prob = model.predict_proba(new_data)[0][1]

st.subheader(f"📈 Predicted Priority for {selected_bus}")
st.metric(label="Maintenance Probability", value=f"{new_prob:.2%}", delta=f"{new_prob - default_row['Predicted']:.2%}")

# إنشاء نسخة قابلة للتعديل من البيانات الأصلية
updated_data = full_data.copy()

# تحديث بيانات الباص المحدد في جميع الأيام
updated_data.loc[updated_data['BusID'] == selected_bus,
              ['EngineTemp', 'OilPressure', 'RPM', 'ErrorCode', 'KM_Today']] = [temp, oil, rpm, encoded_error, km]

# إعادة التنبؤ لكامل البيانات بعد التعديل
X_updated = updated_data[['EngineTemp', 'OilPressure', 'RPM', 'ErrorCode', 'KM_Today']]
updated_data['Predicted'] = model.predict_proba(X_updated)[:, 1]
updated_data['Scheduled'] = updated_data['Predicted'] > 0.7

# إعادة جدولة الباصات: 10 فقط يوميًا
scheduled_data = updated_data[updated_data['Scheduled']].copy()
scheduled_data = full_data[full_data['Scheduled']].copy()
scheduled_data = scheduled_data.sort_values(by=['Date', 'Predicted'], ascending=[True, False])
scheduled_data['DailyCount'] = scheduled_data.groupby('Date').cumcount() + 1
scheduled_data = scheduled_data[scheduled_data['DailyCount'] <= 10]

# إعادة حساب عدد الدخول للكراج

# إشعار إذا تم استبعاد الباص الحالي من الجدول النهائي
if selected_bus not in scheduled_data['BusID'].values:
    st.warning(f"🚨 Bus {selected_bus} has high priority but was not scheduled (other buses had higher priority on those days).")
garage_counts = scheduled_data.groupby('BusID').size().reset_index(name='GarageEntries')

# ---- Bar chart for garage entries ----
st.subheader("📊 Garage Entry Counts for All Buses")
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(garage_counts['BusID'], garage_counts['GarageEntries'], color='skyblue')
ax.set_title("Number of Times Each Bus Enters Garage (Max 10/day)")
ax.set_xlabel("Bus ID")
ax.set_ylabel("Garage Entries")
ax.set_xticks(range(len(garage_counts['BusID'])))
ax.set_xticklabels(garage_counts['BusID'], rotation=90, fontsize=8)
st.pyplot(fig)

# ---- Gantt Chart for Garage Scheduling ----
st.subheader("📅 Gantt Chart: Garage Schedule (Max 10 buses/day)")
scheduled_data['Duration'] = pd.to_timedelta(1, unit='D')
fig, ax = plt.subplots(figsize=(14, 6))
colors = plt.cm.Reds(scheduled_data['Predicted'] / scheduled_data['Predicted'].max())
for i, (bus_id, row) in enumerate(scheduled_data.iterrows()):
    ax.barh(row['BusID'], row['Duration'].days, left=row['Date'], height=0.5, color=colors[i])
ax.set_xlabel("Date")
ax.set_ylabel("Bus ID")
ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.xticks(rotation=45)
plt.title("Gantt Chart of Scheduled Maintenance Events (max 10 buses/day)")
st.pyplot(fig)

st.caption("")
