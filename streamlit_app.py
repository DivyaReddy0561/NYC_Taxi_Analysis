# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score

# Page config
st.set_page_config(page_title="NYC Taxi Analysis", layout="wide")
st.title("🚖 NYC Green Taxi Trip Analysis & Machine Learning")

# File uploader
uploaded_file = st.file_uploader("Upload NYC taxi dataset (Parquet or CSV)", type=["parquet", "csv"])

if uploaded_file:
    try:
        # Load dataset
        if uploaded_file.name.endswith(".parquet"):
            df = pd.read_parquet(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)

        # Initial Display
        st.subheader("🔍 Raw Data Sample")
        st.dataframe(df.head())

        # Drop unused column
        if "ehail_fee" in df.columns:
            df.drop("ehail_fee", axis=1, inplace=True)

        # Feature Engineering
        df["trip_duration"] = (df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]).dt.total_seconds() / 60
        df["weekday"] = df["lpep_dropoff_datetime"].dt.dayofweek
        df["hour"] = df["lpep_dropoff_datetime"].dt.hour
        df = df[df["trip_duration"].between(1, 120)]

        # Fill missing values
        cat_cols = ['store_and_fwd_flag', 'RatecodeID', 'payment_type', 'trip_type']
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mode()[0])
        for col in df.select_dtypes(include=np.number).columns:
            df[col] = df[col].fillna(df[col].median())

        # Encode categorical
        encode_cols = ['store_and_fwd_flag', 'RatecodeID', 'payment_type', 'trip_type', 'PULocationID', 'DOLocationID']
        for col in encode_cols:
            if col in df.columns:
                df[col] = LabelEncoder().fit_transform(df[col])

        st.subheader("🧹 Cleaned Data Preview")
        st.dataframe(df.head())

        # -------- Visualizations --------
        st.subheader("📊 Visualizations")

        col1, col2 = st.columns(2)

        with col1:
            fig1, ax1 = plt.subplots()
            sns.histplot(df["trip_duration"], bins=40, kde=True, color='skyblue', ax=ax1)
            ax1.set_title("Trip Duration Distribution")
            ax1.set_xlabel("Duration (minutes)")
            st.pyplot(fig1)

        with col2:
            fig2, ax2 = plt.subplots()
            sns.scatterplot(x="trip_distance", y="trip_duration", data=df, alpha=0.5, ax=ax2)
            ax2.set_title("Trip Distance vs Duration")
            ax2.set_xlabel("Distance (miles)")
            ax2.set_ylabel("Duration (minutes)")
            st.pyplot(fig2)

        fig3, ax3 = plt.subplots()
        sns.boxplot(x="payment_type", y="fare_amount", data=df, ax=ax3)
        ax3.set_title("Fare Amount by Payment Type")
        st.pyplot(fig3)

        # -------- Modeling --------
        st.subheader("🤖 Machine Learning Models")

        features = ['passenger_count', 'trip_distance', 'PULocationID', 'DOLocationID',
                    'RatecodeID', 'payment_type', 'fare_amount', 'extra', 'mta_tax',
                    'tip_amount', 'tolls_amount', 'improvement_surcharge',
                    'total_amount', 'trip_type', 'weekday', 'hour']

        if not set(features).issubset(df.columns):
            st.error("Missing required features in dataset")
        else:
            X = df[features]
            y = df["trip_duration"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(max_depth=5, min_samples_leaf=10, random_state=42),
                "Random Forest": RandomForestRegressor(n_estimators=50, max_depth=6, min_samples_leaf=10, random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=4, random_state=42)
            }

            results = {}

            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                results[name] = r2

            st.subheader("📈 Model Performance (R² Scores)")
            for model_name, r2 in results.items():
                st.write(f"**{model_name}**: R² = {r2:.3f}")

            # Sample predictions
            st.subheader("🔍 Sample Predictions")
            sample_df = pd.DataFrame({
                "Actual Duration": y_test.iloc[:5].values,
                "Linear Regression": models["Linear Regression"].predict(X_test.iloc[:5]),
                "Decision Tree": models["Decision Tree"].predict(X_test.iloc[:5]),
                "Random Forest": models["Random Forest"].predict(X_test.iloc[:5]),
                "Gradient Boosting": models["Gradient Boosting"].predict(X_test.iloc[:5]),
            })
            st.dataframe(sample_df.round(2))

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
else:
    st.info("📁 Please upload a `.parquet` or `.csv` file to start.")
