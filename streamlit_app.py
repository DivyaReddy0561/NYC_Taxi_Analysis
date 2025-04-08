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

# ---------- MODEL UTILS ----------

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
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        results[name] = r2
        predictions[name] = y_pred
        trained_models[name] = model

    return results, predictions, trained_models

# ---------- STREAMLIT APP ----------

st.set_page_config(page_title="NYC Taxi Analysis & Fare Prediction", layout="wide")
st.title("🗽 NYC Taxi Fare Analysis & Total Amount Prediction")

uploaded_file = st.file_uploader("Upload NYC Taxi Dataset (Parquet or CSV)", type=["parquet", "csv"])

if uploaded_file is not None:
    try:
        # Load dataset
        df = pd.read_parquet(uploaded_file) if uploaded_file.name.endswith(".parquet") else pd.read_csv(uploaded_file)

        st.subheader("Raw Data Sample")
        st.dataframe(df.head())

        # Drop unneeded column
        if 'ehail_fee' in df.columns:
            df.drop("ehail_fee", axis=1, inplace=True)

        # Datetime conversion
        df["lpep_dropoff_datetime"] = pd.to_datetime(df["lpep_dropoff_datetime"])
        df["lpep_pickup_datetime"] = pd.to_datetime(df["lpep_pickup_datetime"])

        # Derived Features
        df["trip_duration"] = (df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]).dt.total_seconds() / 60
        df["weekday"] = df["lpep_pickup_datetime"].dt.dayofweek
        df["hour"] = df["lpep_pickup_datetime"].dt.hour

        df = df[df["trip_duration"].between(1, 120)]  # remove outliers

        # Fill missing values
        cat_cols = ['store_and_fwd_flag', 'RatecodeID', 'payment_type', 'trip_type']
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mode()[0])

        num_cols = df.select_dtypes(include=np.number).columns
        for col in num_cols:
            df[col] = df[col].fillna(df[col].median())

        # Encoding
        encoder_cols = ['store_and_fwd_flag', 'RatecodeID', 'payment_type', 'trip_type', 'PULocationID', 'DOLocationID']
        for col in encoder_cols:
            if col in df.columns:
                df[col] = LabelEncoder().fit_transform(df[col])

        # ---------- VISUALIZATIONS ----------
        st.subheader("🔍 Exploratory Visualizations")

        # Histogram - Trip Distance
        st.markdown("**Trip Distance Distribution**")
        fig1, ax1 = plt.subplots()
        sns.histplot(df["trip_distance"], bins=50, kde=True, ax=ax1)
        ax1.set_xlabel("Trip Distance (miles)")
        st.pyplot(fig1)

        # Bar Plot - Trips by Hour
        st.markdown("**Trip Counts by Hour**")
        fig2, ax2 = plt.subplots()
        sns.countplot(x="hour", data=df, ax=ax2)
        ax2.set_xlabel("Hour of Day")
        ax2.set_ylabel("Trip Count")
        st.pyplot(fig2)

        # Bar Plot - Weekday vs Total Amount
        st.markdown("**Weekday vs Total Amount**")
        fig3, ax3 = plt.subplots()
        weekday_total = df.groupby("weekday")["total_amount"].mean()
        weekday_total.plot(kind="bar", ax=ax3)
        ax3.set_ylabel("Average Total Amount")
        ax3.set_xlabel("Weekday (0=Mon, 6=Sun)")
        st.pyplot(fig3)

        # ---------- MODELING ----------
        st.subheader("🧠 Model Training: Predicting Total Fare Amount")

        features = ['passenger_count', 'trip_distance', 'PULocationID', 'DOLocationID',
                    'RatecodeID', 'payment_type', 'fare_amount', 'extra', 'mta_tax',
                    'tip_amount', 'tolls_amount', 'improvement_surcharge',
                    'trip_type', 'weekday', 'hour']

        if not set(features).issubset(df.columns):
            st.error("Missing required features in dataset.")
        else:
            X = df[features]
            y = df['total_amount']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

            results, predictions, trained_models = train_and_evaluate_models(X_train, X_test, y_train, y_test)

            for name, r2 in results.items():
                st.write(f"**{name} R² score:** {r2:.3f}")

            # Sample predictions
            st.subheader("📊 Sample Predictions (First 5)")
            pred_df = pd.DataFrame({
                "Actual": y_test.iloc[:5].values
            })
            for name, preds in predictions.items():
                pred_df[name] = preds[:5]
            st.dataframe(pred_df)

            # ---------- PREDICTION INTERFACE ----------
            st.subheader("🧮 Predict Total Amount")

            model_name = st.selectbox("Choose model for prediction", list(trained_models.keys()))
            model = trained_models[model_name]

            input_data = {}
            for col in features:
                val = st.number_input(f"{col}", value=float(X[col].median()))
                input_data[col] = val

            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)[0]
            st.success(f"Predicted Total Amount (USD): **${prediction:.2f}**")

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("📁 Please upload a dataset (CSV or Parquet) to begin.")
