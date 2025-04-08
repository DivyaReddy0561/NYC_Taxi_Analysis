# streamlit_app.py

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

# Configure page
st.set_page_config(page_title="NYC Taxi Analysis", layout="wide")
st.title("🚖 NYC Green Taxi Trip Data Analysis & ML Modeling")

# File uploader
uploaded_file = st.file_uploader("📁 Upload a dataset (Parquet or CSV)", type=["parquet", "csv"])

if uploaded_file is not None:
    try:
        # Load dataset
        if uploaded_file.name.endswith('.parquet'):
            df = pd.read_parquet(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)

        st.subheader("🧾 Raw Data Sample")
        st.dataframe(df.head())

        # Parse datetime columns
        df["lpep_pickup_datetime"] = pd.to_datetime(df["lpep_pickup_datetime"], errors='coerce')
        df["lpep_dropoff_datetime"] = pd.to_datetime(df["lpep_dropoff_datetime"], errors='coerce')

        # Drop unneeded columns
        if 'ehail_fee' in df.columns:
            df.drop("ehail_fee", axis=1, inplace=True)

        # Feature Engineering
        df["trip_duration"] = (df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]).dt.total_seconds() / 60
        df["weekday"] = df["lpep_dropoff_datetime"].dt.dayofweek
        df["hour"] = df["lpep_dropoff_datetime"].dt.hour

        # Filter out unrealistic durations
        df = df[df["trip_duration"].between(1, 120)]

        # Handle missing categorical values
        cat_cols = ['store_and_fwd_flag', 'RatecodeID', 'payment_type', 'trip_type']
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mode()[0])

        # Handle missing numeric values
        num_cols = df.select_dtypes(include=np.number).columns
        for col in num_cols:
            df[col] = df[col].fillna(df[col].median())

        # Encode categorical variables
        encoder_cols = ['store_and_fwd_flag', 'RatecodeID', 'payment_type', 'trip_type', 'PULocationID', 'DOLocationID']
        for col in encoder_cols:
            if col in df.columns:
                df[col] = LabelEncoder().fit_transform(df[col])

        st.subheader("✅ Processed Dataset Preview")
        st.dataframe(df.head())

        # Feature list (excluding 'total_amount' to avoid misleading model)
        features = ['passenger_count', 'trip_distance', 'PULocationID', 'DOLocationID',
                    'RatecodeID', 'payment_type', 'fare_amount', 'extra', 'mta_tax',
                    'tip_amount', 'tolls_amount', 'improvement_surcharge',
                    'trip_type', 'weekday', 'hour']

        # Check for missing required features
        if not set(features).issubset(df.columns):
            st.error("❌ Required features are missing in the dataset.")
        else:
            X = df[features]
            y = df['trip_duration']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

            # Models
            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(max_depth=5, min_samples_leaf=10, random_state=42),
                "Random Forest": RandomForestRegressor(n_estimators=50, max_depth=6, min_samples_leaf=10, random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=4, random_state=42)
            }

            st.subheader("📊 Regression Model R² Scores")
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                st.write(f"**{name}** R² Score: **{r2:.3f}**")

            # Show sample predictions
            st.subheader("🔍 Sample Predictions (Random 5 rows)")
            sample = X_test.sample(5, random_state=42)
            sample_preds = pd.DataFrame({
                "Actual (minutes)": y_test.loc[sample.index].values,
                "Linear Regression": models["Linear Regression"].predict(sample),
                "Decision Tree": models["Decision Tree"].predict(sample),
                "Random Forest": models["Random Forest"].predict(sample),
                "Gradient Boosting": models["Gradient Boosting"].predict(sample),
            })
            st.dataframe(sample_preds)

    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")

else:
    st.info("👆 Please upload a Parquet or CSV file to begin.")
