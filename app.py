from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import csv
import os

app = Flask(__name__)

# ============================================================
# FILE PATHS
# ============================================================

MODEL_FILE = "battery_soh_model.pkl"
CSV_FILE = "battery_features.csv"


# ============================================================
# LOAD MODEL
# ============================================================

try:
    model = joblib.load(MODEL_FILE)
    print("Model loaded successfully.")
except Exception as e:
    print("ERROR loading model:", e)
    model = None


# ============================================================
# LOAD CSV
# ============================================================

def load_battery_data():

    if not os.path.exists(CSV_FILE):
        print("WARNING: battery_features.csv not found.")
        return pd.DataFrame()

    try:
        # First try normal pandas reading
        df = pd.read_csv(CSV_FILE)

        # Remove unwanted spaces from column names
        df.columns = [
            str(col).strip().replace("\ufeff", "")
            for col in df.columns
        ]

        # ----------------------------------------------------
        # If pandas sees only ONE column, try Python csv reader
        # ----------------------------------------------------

        if len(df.columns) == 1:

            print("Pandas detected only one CSV column.")
            print("Trying alternative CSV reader...")

            with open(
                CSV_FILE,
                "r",
                encoding="utf-8-sig",
                newline=""
            ) as file:

                reader = csv.reader(file)

                rows = list(reader)

            if len(rows) > 0:

                headers = [
                    str(x).strip()
                    for x in rows[0]
                ]

                data_rows = rows[1:]

                df = pd.DataFrame(
                    data_rows,
                    columns=headers
                )

        # Clean column names again
        df.columns = [
            str(col).strip().replace("\ufeff", "")
            for col in df.columns
        ]

        print("CSV columns:")
        print(list(df.columns))

        return df

    except Exception as e:

        print("ERROR loading CSV:", e)

        # ----------------------------------------------------
        # Final fallback using csv module
        # ----------------------------------------------------

        try:

            with open(
                CSV_FILE,
                "r",
                encoding="utf-8-sig",
                newline=""
            ) as file:

                reader = csv.reader(file)
                rows = list(reader)

            if len(rows) < 2:
                return pd.DataFrame()

            headers = [
                str(x).strip()
                for x in rows[0]
            ]

            data = rows[1:]

            df = pd.DataFrame(
                data,
                columns=headers
            )

            print("CSV loaded using fallback reader.")
            print("CSV columns:")
            print(list(df.columns))

            return df

        except Exception as e2:

            print("Fallback CSV loading also failed:")
            print(e2)

            return pd.DataFrame()


# Load CSV
df = load_battery_data()


# ============================================================
# FIND CYCLE AND CAPACITY COLUMNS
# ============================================================

def find_column(dataframe, possible_names):

    if dataframe.empty:
        return None

    # Create normalized names
    normalized_columns = {}

    for column in dataframe.columns:

        clean_name = (
            str(column)
            .strip()
            .lower()
            .replace(" ", "")
            .replace("_", "")
            .replace("-", "")
        )

        normalized_columns[clean_name] = column

    for name in possible_names:

        normalized_name = (
            name
            .strip()
            .lower()
            .replace(" ", "")
            .replace("_", "")
            .replace("-", "")
        )

        if normalized_name in normalized_columns:

            return normalized_columns[normalized_name]

    return None


# Find cycle column
cycle_column = find_column(
    df,
    [
        "Cycle",
        "cycle",
        "cycle_number",
        "cycle number",
        "cycles"
    ]
)


# Find capacity column
capacity_column = find_column(
    df,
    [
        "Capacity_mAh",
        "capacity_mAh",
        "Capacity",
        "capacity",
        "capacity mah",
        "capacitymah"
    ]
)


print("Detected cycle column:", cycle_column)
print("Detected capacity column:", capacity_column)


# ============================================================
# PREPARE CSV DATA
# ============================================================

if not df.empty:

    # Convert cycle to numeric if found
    if cycle_column is not None:

        df[cycle_column] = pd.to_numeric(
            df[cycle_column],
            errors="coerce"
        )

    # Convert capacity to numeric if found
    if capacity_column is not None:

        df[capacity_column] = pd.to_numeric(
            df[capacity_column],
            errors="coerce"
        )


# ============================================================
# DISPLAY STARTUP INFORMATION
# ============================================================

print()
print("==============================================")
print(" BATTERY SOH PREDICTION APP")
print("==============================================")
print("CSV file:", CSV_FILE)
print("Model file:", MODEL_FILE)
print("Cycle column:", cycle_column)
print("Capacity column:", capacity_column)
print("==============================================")
print()


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template("index.html")


# ============================================================
# PREDICTION
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # ----------------------------------------------------
        # Check model
        # ----------------------------------------------------

        if model is None:

            return jsonify({
                "error": "Battery SOH model could not be loaded."
            }), 500


        # ----------------------------------------------------
        # Get JSON data from webpage
        # ----------------------------------------------------

        data = request.get_json()

        if data is None:

            return jsonify({
                "error": "No input data received."
            }), 400


        # ----------------------------------------------------
        # Get cycle
        # ----------------------------------------------------

        if "cycle" not in data:

            return jsonify({
                "error": "Cycle value is missing."
            }), 400


        # ----------------------------------------------------
        # Get capacity
        # ----------------------------------------------------

        if "capacity" not in data:

            return jsonify({
                "error": "Capacity value is missing."
            }), 400


        cycle = float(data["cycle"])
        capacity = float(data["capacity"])


        # ----------------------------------------------------
        # Check values
        # ----------------------------------------------------

        if cycle < 0:

            return jsonify({
                "error": "Cycle cannot be negative."
            }), 400


        if capacity <= 0:

            return jsonify({
                "error": "Capacity must be greater than zero."
            }), 400


        MAX_CAPACITY = 724.13

        if capacity > MAX_CAPACITY:

            return jsonify({
                "error": f"Capacity is outside the supported range. Please enter a value between 0 and {MAX_CAPACITY:.2f}."
            }), 400
       # SOH PREDICTION
       #
       # ================================================

        INITIAL_CAPACITY = 724.13

        soh = round(
               (capacity / INITIAL_CAPACITY) * 
       100,
              2
      )


        # ====================================================
        # FIND CLOSEST BATTERY DATA
        # ====================================================

        battery_data = {

            "cycle": cycle,

            "capacity": round(
                capacity,
                2
            ),

            "time_duration": None,

            "voltage_mean": None,

            "voltage_min": None,

            "voltage_max": None,

            "charge_mean": None,

            "charge_max": None,

            "temperature_mean": None,

            "temperature_max": None
        }


        # ----------------------------------------------------
        # Only search CSV if both columns were found
        # ----------------------------------------------------

        if (
            not df.empty
            and cycle_column is not None
            and capacity_column is not None
        ):

            # Remove rows where cycle/capacity are missing
            valid_df = df.dropna(
                subset=[
                    cycle_column,
                    capacity_column
                ]
            ).copy()


            if not valid_df.empty:

                # ------------------------------------------------
                # Calculate distance from entered values
                # ------------------------------------------------

                valid_df["difference"] = (

                    abs(
                        valid_df[cycle_column] - cycle
                    )

                    +

                    abs(
                        valid_df[capacity_column] - capacity
                    )
                )


                # Get closest row
                closest_index = (
                    valid_df["difference"]
                    .idxmin()
                )

                row = valid_df.loc[
                    closest_index
                ]


                # ------------------------------------------------
                # Helper function
                # ------------------------------------------------

                def get_value(column_name):

                    if column_name in valid_df.columns:

                        try:

                            value = float(
                                row[column_name]
                            )

                            if pd.isna(value):
                                return None

                            return round(
                                value,
                                6
                            )

                        except Exception:

                            return None

                    return None


                # ------------------------------------------------
                # Battery information
                # ------------------------------------------------

                battery_data["cycle"] = round(
                    float(row[cycle_column]),
                    2
                )

                battery_data["capacity"] = round(
                    float(row[capacity_column]),
                    2
                )


                # ------------------------------------------------
                # Other battery parameters
                # ------------------------------------------------

                battery_data["time_duration"] = get_value(
                    "time_duration"
                )

                battery_data["voltage_mean"] = get_value(
                    "voltage_mean"
                )

                battery_data["voltage_min"] = get_value(
                    "voltage_min"
                )

                battery_data["voltage_max"] = get_value(
                    "voltage_max"
                )

                battery_data["charge_mean"] = get_value(
                    "charge_mean"
                )

                battery_data["charge_max"] = get_value(
                    "charge_max"
                )

                battery_data["temperature_mean"] = get_value(
                    "temperature_mean"
                )

                battery_data["temperature_max"] = get_value(
                    "temperature_max"
                )


        # ====================================================
        # RETURN RESULT
        # ====================================================

        return jsonify({

            "prediction": soh,

            "battery_data": battery_data

        })


    except Exception as e:

        print("Prediction error:", e)

        return jsonify({

            "error": str(e)

        }), 500


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return jsonify({

        "status": "running",

        "model_loaded": model is not None,

        "csv_loaded": not df.empty,

        "cycle_column": cycle_column,

        "capacity_column": capacity_column

    })


# ============================================================
# RUN FLASK
# ============================================================

if __name__ == "__main__":

    print("Starting Flask server...")
    print("Open: http://127.0.0.1:5000")
    print()

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
