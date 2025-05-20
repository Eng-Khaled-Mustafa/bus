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

# ---- Predict initial probabilities before controls ----
full_data['Predicted'] = model.predict_proba(
    full_data[['EngineTemp', 'OilPressure', 'RPM', 'ErrorCode', 'KM_Today']]
)[:, 1]

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
new_prob = round(
    0.35 * (temp / 120) +
    0.35 * (1 - oil / 5) +
    0.15 * (rpm / 2500) +
    0.15 * (1 if error != 'None' else 0), 4
) +
    0.3 * (1 - oil / 5) +
    0.2 * (rpm / 2500) +
    0.2 * (1 if error != 'None' else 0)
)

st.subheader(f"📈 Predicted Priority for {selected_bus}")
st.metric(label="Maintenance Probability", value=f"{new_prob:.2%}", delta=f"{new_prob - default_row['Predicted']:.2%}")

# ---- Apply changes to selected bus for all days ----
updated_data = full_data.copy()
updated_data.loc[updated_data['BusID'] == selected_bus,
              ['EngineTemp', 'OilPressure', 'RPM', 'ErrorCode', 'KM_Today']] = [temp, oil, rpm, encoded_error, km]

# Predict on updated data
X_updated = updated_data[['EngineTemp', 'OilPressure', 'RPM', 'ErrorCode', 'KM_Today']]
updated_data['Predicted'] = round(
    0.35 * (updated_data['EngineTemp'] / 120) +
    0.35 * (1 - updated_data['OilPressure'] / 5) +
    0.15 * (updated_data['RPM'] / 2500) +
    0.15 * (updated_data['ErrorCode'] != le.transform(['None'])[0]).astype(float), 4
) +
    0.3 * (1 - updated_data['OilPressure'] / 5) +
    0.2 * (updated_data['RPM'] / 2500) +
    0.2 * (updated_data['ErrorCode'] != le.transform(['None'])[0]).astype(float)
)
scheduled_mask = (updated_data['Predicted'] > 0.7) & (np.random.rand(len(updated_data)) < 0.3)
updated_data['Scheduled'] = scheduled_mask

# Schedule top 10 buses per day
scheduled_data = updated_data[updated_data['Scheduled']].copy()
scheduled_data = scheduled_data.sort_values(by=['Date', 'Predicted'], ascending=[True, False])
scheduled_data['DailyCount'] = scheduled_data.groupby('Date').cumcount() + 1
scheduled_data = scheduled_data[scheduled_data['DailyCount'] <= 10]

# Garage entry count
garage_counts = scheduled_data.groupby('BusID').size().reset_index(name='GarageEntries')

# Show warning if selected bus was excluded
total_entries = garage_counts[garage_counts['BusID'] == selected_bus]['GarageEntries'].sum()
if total_entries == 0:
    st.warning(f"🚨 Bus {selected_bus} has high priority but was not scheduled (other buses had higher priority on those days).")

# ---- Bar chart ----
st.subheader("📊 Garage Entry Counts for All Buses")
fig, ax = plt.subplots(figsize=(12, 6))
colors = plt.cm.tab20.colors
bar_colors = [colors[i % len(colors)] for i in range(len(garage_counts))]
ax.bar(garage_counts['BusID'], garage_counts['GarageEntries'], color=bar_colors)
ax.set_title("Number of Times Each Bus Enters Garage (Max 10/day)")
ax.set_xlabel("Bus ID")
ax.set_ylabel("Garage Entries")
ax.set_xticks(range(len(garage_counts['BusID'])))
ax.set_xticklabels(garage_counts['BusID'], rotation=90, fontsize=8)
st.pyplot(fig)

# ---- Gantt Chart ----
st.subheader("📅 Gantt Chart: Garage Schedule (Max 10 buses/day)")
scheduled_data['Duration'] = pd.to_timedelta(1, unit='D')
fig, ax = plt.subplots(figsize=(14, 6))

# استخدم colormap ملون بناء على الاحتمالية
norm = plt.Normalize(scheduled_data['Predicted'].min(), scheduled_data['Predicted'].max())
colors = plt.cm.viridis(norm(scheduled_data['Predicted']))

for i, (bus_id, row) in enumerate(scheduled_data.iterrows()):
    ax.barh(row['BusID'], row['Duration'].days, left=row['Date'], height=0.5, color=colors[i])

ax.set_xlabel("Date")
ax.set_ylabel("Bus ID")
ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.xticks(rotation=45)
plt.title("Gantt Chart of Scheduled Maintenance Events (max 10 buses/day)")
st.pyplot(fig)

st.caption("تم احترام سعة الكراج بحيث لا تتجاوز 10 باصات يوميًا بناءً على ترتيب الأولوية والاحتمالية.")
