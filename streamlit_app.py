import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score

# ---------- MODEL UTILS (Encapsulated) ----------

def get_models():
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=5, min_samples_leaf=10, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=50, max_depth=6, min_samples_leaf=10, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=4, random_state=42)
    }

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = get_models()
    results = {}
    predictions = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        results[name] = r2
        predictions[name] = y_pred

    return results, predictions, models

# ---------- STREAMLIT APP ----------

st.set_page_config(page_title="NYC Taxi Analysis", layout="wide")
st.title("NYC Green Taxi Trip Data Analysis & ML Modeling")

uploaded_file = st.file_uploader("Upload a dataset (Parquet or CSV)", type=["parquet", "csv"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.parquet'):
            df = pd.read_parquet(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)

        st.subheader("Raw Data Sample")
        st.dataframe(df.head())

        if 'ehail_fee' in df.columns:
            df.drop("ehail_fee", axis=1, inplace=True)

        df["lpep_dropoff_datetime"] = pd.to_datetime(df["lpep_dropoff_datetime"])
        df["lpep_pickup_datetime"] = pd.to_datetime(df["lpep_pickup_datetime"])

        df["trip_duration"] = (df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]).dt.total_seconds() / 60
        df["weekday"] = df["lpep_dropoff_datetime"].dt.dayofweek
        df["hour"] = df["lpep_dropoff_datetime"].dt.hour

        df = df[df["trip_duration"].between(1, 120)]

        cat_cols = ['store_and_fwd_flag', 'RatecodeID', 'payment_type', 'trip_type']
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mode()[0])

        num_cols = df.select_dtypes(include=np.number).columns
        for col in num_cols:
            df[col] = df[col].fillna(df[col].median())

        encoder_cols = ['store_and_fwd_flag', 'RatecodeID', 'payment_type', 'trip_type', 'PULocationID', 'DOLocationID']
        for col in encoder_cols:
            if col in df.columns:
                df[col] = LabelEncoder().fit_transform(df[col])

        st.subheader("Processed Dataset")
        st.dataframe(df.head())

        features = ['passenger_count', 'trip_distance', 'PULocationID', 'DOLocationID',
                    'RatecodeID', 'payment_type', 'fare_amount', 'extra', 'mta_tax',
                    'tip_amount', 'tolls_amount', 'improvement_surcharge',
                    'total_amount', 'trip_type', 'weekday', 'hour']
        
        if not set(features).issubset(df.columns):
            st.error("Missing required features in dataset")
        else:
            X = df[features]
            y = df['trip_duration']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

            st.subheader("Regression Model R² Scores")
            results, predictions, trained_models = train_and_evaluate_models(X_train, X_test, y_train, y_test)

            for name, r2 in results.items():
                st.write(f"{name} R² score: **{r2:.3f}**")

            st.subheader("Sample Predictions (First 5)")
            sample_preds = pd.DataFrame({"Actual": y_test.iloc[:5].values})
            for name, preds in predictions.items():
                sample_preds[name] = preds[:5]
            st.dataframe(sample_preds)

    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    st.info("📁 Please upload a Parquet or CSV file to begin.")
