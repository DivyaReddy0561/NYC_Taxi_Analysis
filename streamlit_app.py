# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score

# Configure page
st.set_page_config(page_title="NYC Taxi Analysis", layout="wide")
st.title("🚖 NYC Green Taxi Trip Duration Prediction")

uploaded_file = st.file_uploader("Upload NYC Green Taxi dataset (.parquet or .csv)", type=["parquet", "csv"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.parquet'):
        df = pd.read_parquet(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    st.subheader("🔍 Raw Data Sample")
    st.dataframe(df.head())

    # Drop unwanted columns
    if 'ehail_fee' in df.columns:
        df.drop("ehail_fee", axis=1, inplace=True)

    # Feature Engineering
    df["lpep_dropoff_datetime"] = pd.to_datetime(df["lpep_dropoff_datetime"])
    df["lpep_pickup_datetime"] = pd.to_datetime(df["lpep_pickup_datetime"])
    df["trip_duration"] = (df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]).dt.total_seconds() / 60
    df = df[df["trip_duration"].between(1, 120)]

    df["weekday"] = df["lpep_dropoff_datetime"].dt.dayofweek
    df["hour"] = df["lpep_dropoff_datetime"].dt.hour

    # Handle missing values
    cat_cols = ['store_and_fwd_flag', 'RatecodeID', 'payment_type', 'trip_type']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    num_cols = df.select_dtypes(include=np.number).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    # Encode categoricals
    encode_cols = ['store_and_fwd_flag', 'RatecodeID', 'payment_type', 'trip_type', 'PULocationID', 'DOLocationID']
    for col in encode_cols:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col])

    st.subheader("🛠️ Processed Data Sample")
    st.dataframe(df.head())

    # Features for modeling
    features = ['passenger_count', 'trip_distance', 'PULocationID', 'DOLocationID',
                'RatecodeID', 'payment_type', 'fare_amount', 'extra', 'mta_tax',
                'tip_amount', 'tolls_amount', 'improvement_surcharge',
                'trip_type', 'weekday', 'hour']

    if not set(features).issubset(df.columns):
        st.error("⚠️ Dataset missing required features.")
    else:
        X = df[features]
        y = df['trip_duration']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        # Initialize models
        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(max_depth=5, min_samples_leaf=10, random_state=42),
            "Random Forest": RandomForestRegressor(n_estimators=50, max_depth=6, min_samples_leaf=10, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=4, random_state=42)
        }

        # Train and evaluate
        st.subheader("📈 Model Performance (R² Scores)")
        model_scores = {}
        trained_models = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            score = r2_score(y_test, preds)
            model_scores[name] = score
            trained_models[name] = model
            st.write(f"**{name}** R² score: `{score:.3f}`")

        # Select model
        st.subheader("🤖 Select Model for Prediction")
        selected_model_name = st.selectbox("Choose model", list(trained_models.keys()))
        selected_model = trained_models[selected_model_name]
        st.markdown(f"**Selected Model R² Score:** `{model_scores[selected_model_name]:.3f}`")

        # Input form
        st.subheader("✏️ Enter Ride Details to Predict Trip Duration")
        user_input = {}
        for col in features:
            if col in ['passenger_count', 'PULocationID', 'DOLocationID', 'RatecodeID', 'payment_type', 'trip_type', 'weekday', 'hour']:
                val = st.number_input(f"{col}", min_value=0, value=int(X[col].median()), step=1)
            else:
                val = st.number_input(f"{col}", value=float(round(X[col].median(), 2)))
            user_input[col] = val

        if st.button("📍 Predict Trip Duration"):
            input_df = pd.DataFrame([user_input])
            predicted_duration = selected_model.predict(input_df)[0]
            st.success(f"⏱️ Estimated Trip Duration: **{predicted_duration:.2f} minutes**")
