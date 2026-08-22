import streamlit as st
import pandas as pd
import os

# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="EV Battery Health",
    page_icon="🔋",
    layout="centered"
)

# ============================================================
# NEON DESIGN
# ============================================================

st.markdown("""
<style>
.stApp {
    background-color: #050505;
    color: white;
}

h1 {
    color: #00ffff !important;
    text-align: center;
    text-shadow: 0 0 10px #00ffff;
}

h2 {
    color: #00ffff !important;
    text-shadow: 0 0 8px #00ffff;
}

p {
    color: white !important;
}

div[data-testid="stNumberInput"] label {
    color: #00ffff !important;
}

div[data-testid="stNumberInput"] input {
    background-color: #111111;
    color: #00ffff;
    border: 1px solid #00ffff;
    border-radius: 8px;
}

.stButton > button {
    background-color: #ff00ff;
    color: white;
    border: 1px solid #ff00ff;
    border-radius: 10px;
    box-shadow: 0 0 10px #ff00ff;
    font-weight: bold;
}

.stButton > button:hover {
    background-color: #00ffff;
    color: black;
    box-shadow: 0 0 15px #00ffff;
}

.result-box {
    background-color: #111111;
    border: 1px solid #00ffff;
    border-radius: 12px;
    padding: 20px;
    margin-top: 20px;
    box-shadow: 0 0 15px rgba(0,255,255,0.4);
}

.soh {
    color: #00ffff;
    font-size: 42px;
    font-weight: bold;
    text-align: center;
    text-shadow: 0 0 10px #00ffff;
}

.parameter {
    color: white;
    font-size: 17px;
    padding: 7px 0;
}

.parameter-name {
    color: #00ffff;
    font-weight: bold;
}

.info-box {
    background-color: #111111;
    border: 1px solid #333333;
    border-radius: 10px;
    padding: 15px;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATASET
# ============================================================

CSV_FILE = "battery_features.csv"

if not os.path.exists(CSV_FILE):
    st.error("❌ battery_features.csv was not found.")
    st.stop()

try:
    df = pd.read_csv(
        CSV_FILE,
        encoding="utf-8-sig"
    )
except Exception as e:
    st.error(f"❌ Error loading dataset: {e}")
    st.stop()


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

df.columns = [
    str(col).strip().replace("\ufeff", "")
    for col in df.columns
]


# ============================================================
# FIND COLUMN FUNCTION
# ============================================================

def find_column(possible_names):

    normalized = {}

    for col in df.columns:

        name = (
            str(col)
            .strip()
            .lower()
            .replace(" ", "")
            .replace("_", "")
            .replace("-", "")
        )

        normalized[name] = col

    for name in possible_names:

        clean = (
            str(name)
            .strip()
            .lower()
            .replace(" ", "")
            .replace("_", "")
            .replace("-", "")
        )

        if clean in normalized:
            return normalized[clean]

    return None


# ============================================================
# FIND IMPORTANT COLUMNS
# ============================================================

cycle_column = find_column([
    "Cycle",
    "cycle_number",
    "cycle number",
    "cycles"
])

capacity_column = find_column([
    "Capacity_mAh",
    "capacity_mAh",
    "Capacity",
    "capacity",
    "capacity mah",
    "capacitymah"
])


# ============================================================
# CHECK COLUMNS
# ============================================================

if cycle_column is None:
    st.error("❌ Cycle column not found in battery_features.csv.")
    st.stop()

if capacity_column is None:
    st.error("❌ Capacity column not found in battery_features.csv.")
    st.stop()


# ============================================================
# TITLE
# ============================================================

st.markdown(
    "<h1>🔋 EV Battery Health Prediction</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;'>Enter the battery details to estimate State of Health (SOH).</p>",
    unsafe_allow_html=True
)


# ============================================================
# INPUTS
# ============================================================

cycle = st.number_input(
    "Battery Cycle",
    min_value=0.0,
    value=500.0,
    step=1.0
)

capacity = st.number_input(
    "Battery Capacity (mAh)",
    min_value=0.1,
    value=500.0,
    step=1.0
)


# ============================================================
# PREDICT
# ============================================================

if st.button("🔍 Predict Battery Health"):

    # ========================================================
    # SOH
    # ========================================================

    INITIAL_CAPACITY = 724.13

    soh = round(
        (capacity / INITIAL_CAPACITY) * 100,
        2
    )


    # ========================================================
    # PREPARE DATA
    # ========================================================

    valid_df = df.copy()

    valid_df[cycle_column] = pd.to_numeric(
        valid_df[cycle_column],
        errors="coerce"
    )

    valid_df[capacity_column] = pd.to_numeric(
        valid_df[capacity_column],
        errors="coerce"
    )

    valid_df = valid_df.dropna(
        subset=[
            cycle_column,
            capacity_column
        ]
    )


    if valid_df.empty:
        st.error("❌ No valid data found in dataset.")
        st.stop()


    # ========================================================
    # FIND CLOSEST RECORD
    # ========================================================

    valid_df["difference"] = (
        abs(valid_df[cycle_column] - cycle)
        +
        abs(valid_df[capacity_column] - capacity)
    )

    closest_index = valid_df["difference"].idxmin()

    row = valid_df.loc[closest_index]


    # ========================================================
    # PREDICTION RESULT
    # ========================================================

    st.markdown(
        "<h2>Prediction Result</h2>",
        unsafe_allow_html=True
    )

    st.markdown(
        f"<div class='result-box'>"
        f"<div style='text-align:center;color:white;font-size:18px;'>Battery SOH</div>"
        f"<div class='soh'>{soh}%</div>"
        f"</div>",
        unsafe_allow_html=True
    )


    # ========================================================
    # BATTERY CONDITION
    # ========================================================

    if soh >= 80:

        st.success("🟢 Battery Condition: GOOD")

    elif soh >= 60:

        st.warning("🟡 Battery Condition: WARNING")

    else:

        st.error("🔴 Battery Condition: CRITICAL")


    # ========================================================
    # BATTERY PARAMETERS
    # ========================================================

    st.markdown(
        "<h2>⚡ Battery Parameters</h2>",
        unsafe_allow_html=True
    )


    # ========================================================
    # FUNCTION TO DISPLAY PARAMETER
    # ========================================================

    def display_parameter(name, value):

        if pd.isna(value):
            value = "N/A"
        else:
            try:
                value = round(float(value), 6)
            except:
                value = str(value)

        st.markdown(
            f"<div class='parameter'>"
            f"<span class='parameter-name'>{name}:</span> {value}"
            f"</div>",
            unsafe_allow_html=True
        )


    # ========================================================
    # CYCLE
    # ========================================================

    display_parameter(
        "Cycle",
        row[cycle_column]
    )


    # ========================================================
    # CAPACITY
    # ========================================================

    display_parameter(
        "Capacity",
        row[capacity_column]
    )


    # ========================================================
    # OTHER PARAMETERS
    # ========================================================

    parameters = {

        "Time Duration": [
            "time_duration",
            "time duration"
        ],

        "Voltage Mean": [
            "voltage_mean",
            "voltage mean"
        ],

        "Voltage Min": [
            "voltage_min",
            "voltage min"
        ],

        "Voltage Max": [
            "voltage_max",
            "voltage max"
        ],

        "Charge Mean": [
            "charge_mean",
            "charge mean"
        ],

        "Charge Max": [
            "charge_max",
            "charge max"
        ],

        "Temperature Mean": [
            "temperature_mean",
            "temperature mean"
        ],

        "Temperature Max": [
            "temperature_max",
            "temperature max"
        ]
    }


    # ========================================================
    # DISPLAY PARAMETERS
    # ========================================================

    for display_name, names in parameters.items():

        column = find_column(names)

        if column is not None:

            display_parameter(
                display_name,
                row[column]
            )


    # ========================================================
    # INFO
    # ========================================================

    st.markdown(
        "<div class='info-box'>"
        "<span style='color:#00ffff;font-weight:bold;'>"
        "Dataset Match:"
        "</span> "
        "The displayed battery parameters are taken from "
        "the closest matching record in your dataset."
        "</div>",
        unsafe_allow_html=True
    )
