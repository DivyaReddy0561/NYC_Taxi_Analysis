# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

# Configure page
st.set_page_config(page_title="NYC Taxi Analysis", layout="wide")
st.title("NYC Green Taxi Trip Data Analysis")

# File uploader
uploaded_file = st.file_uploader("Upload Parquet File", type=["parquet"])

if uploaded_file is not None:
    try:
        # Load data
        nycgreen = pd.read_parquet(uploaded_file)

        # Data preprocessing
        with st.expander("Data Preprocessing Steps", expanded=True):
            if 'ehail_fee' in nycgreen.columns:
                nycgreen = nycgreen.drop("ehail_fee", axis=1)
                st.success("Dropped ehail_fee column")

            # Calculate trip duration
            nycgreen["trip_duration"] = (
                nycgreen["lpep_dropoff_datetime"] - nycgreen["lpep_pickup_datetime"]
            ).dt.total_seconds() / 60

            # Extract datetime features
            nycgreen["weekday"] = nycgreen["lpep_dropoff_datetime"].dt.day_name()
            nycgreen["hour"] = nycgreen["lpep_dropoff_datetime"].dt.hour

            st.write("### Processed Data Preview")
            st.dataframe(nycgreen.head())

        # Missing value handling
        with st.expander("Missing Value Treatment"):
            objcols = ['store_and_fwd_flag', 'RatecodeID', 'payment_type', 'trip_type']
            for col in objcols:
                if col in nycgreen.columns:
                    mode_val = nycgreen[col].mode()[0]
                    nycgreen[col] = nycgreen[col].fillna(mode_val)

            numcols = nycgreen.select_dtypes(include=np.number).columns
            for col in numcols:
                if nycgreen[col].isnull().sum() > 0:
                    median_val = nycgreen[col].median()
                    nycgreen[col] = nycgreen[col].fillna(median_val)

            st.success("Missing values imputed")

        # EDA Section
        st.header("Exploratory Data Analysis")

        with st.expander("Categorical Variable Analysis"):
            col1, col2 = st.columns(2)

            with col1:
                st.write("### Weekday Distribution")
                weekday_counts = nycgreen["weekday"].value_counts()
                fig, ax = plt.subplots()
                ax.pie(weekday_counts, labels=weekday_counts.index, autopct='%1.1f%%')
                st.pyplot(fig)

            with col2:
                st.write("### Hourly Distribution")
                hour_counts = nycgreen["hour"].value_counts().sort_index()
                st.bar_chart(hour_counts)

        # Feature Engineering
        st.header("Feature Engineering")

        with st.expander("Variable Encoding"):
            high_cardinality = ['PULocationID', 'DOLocationID']
            le = LabelEncoder()
            for col in high_cardinality:
                if col in nycgreen.columns:
                    nycgreen[col] = le.fit_transform(nycgreen[col])

            categorical_cols = ['store_and_fwd_flag', 'RatecodeID', 'payment_type', 'trip_type']
            for col in categorical_cols:
                if col in nycgreen.columns:
                    nycgreen[col] = nycgreen[col].astype('category').cat.codes

            st.write("### Encoded Data Preview")
            st.dataframe(nycgreen.head())

        # Final Dataset
        st.header("Processed Dataset")
        st.write("### Final Data Structure")
        st.dataframe(nycgreen.describe())

        # ----------------- MODELING SECTION -------------------
        st.header("Machine Learning: Fare Prediction")

        with st.expander("Train a Model to Predict Fare", expanded=True):
            target_col = "fare_amount"
            if target_col in nycgreen.columns:
                features = nycgreen.drop(columns=[target_col, 'lpep_pickup_datetime', 'lpep_dropoff_datetime'])
                target = nycgreen[target_col]

                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    features, target, test_size=0.2, random_state=42
                )

                # Random Forest Regressor (tuned to reduce overfitting)
                rf_model = RandomForestRegressor(
                    n_estimators=50,
                    max_depth=5,
                    min_samples_leaf=10,
                    random_state=42
                )
                rf_model.fit(X_train, y_train)

                # Evaluation
                train_r2 = rf_model.score(X_train, y_train)
                test_r2 = rf_model.score(X_test, y_test)
                preds = rf_model.predict(X_test)
                mse = mean_squared_error(y_test, preds)

                st.subheader("Model Evaluation")
                st.write(f"Train R²: {train_r2:.3f}")
                st.write(f"Test R²: {test_r2:.3f}")
                st.write(f"Test MSE: {mse:.2f}")

                st.subheader("Sample Predictions")
                sample_df = pd.DataFrame({
                    "Actual": y_test[:5].values,
                    "Predicted": preds[:5]
                })
                st.dataframe(sample_df)
            else:
                st.warning(f"`{target_col}` column not found. Cannot perform regression.")

    except Exception as e:
        st.error(f"Error processing data: {str(e)}")

else:
    st.info("👆 Please upload a Parquet file to begin analysis")
