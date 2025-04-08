# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score

# Config
st.set_page_config(page_title="NYC Taxi EDA & Regression", layout="wide")
st.title("NYC Green Taxi Data Analysis & Modeling")

uploaded_file = st.file_uploader("📂 Upload NYC Green Taxi .parquet File", type="parquet")

if uploaded_file:
    df = pd.read_parquet(uploaded_file)

    st.subheader("a) Dataset Info")
    st.write(df.info())

    # b) Drop ehail_fee
    if 'ehail_fee' in df.columns:
        df.drop("ehail_fee", axis=1, inplace=True)
        st.success("Dropped column: ehail_fee")

    # c) Calculate trip_duration
    df["trip_duration"] = (df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]).dt.total_seconds() / 60

    # d) Weekday extraction
    df["weekday"] = df["lpep_dropoff_datetime"].dt.day_name()
    st.write("d) Weekday Distribution")
    st.write(df["weekday"].value_counts())

    # e) Hour of day extraction
    df["hourofday"] = df["lpep_dropoff_datetime"].dt.hour
    st.write("e) Hourly Distribution")
    st.write(df["hourofday"].value_counts())

    # f) Missing value imputation
    for col in df.select_dtypes(include='object').columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
    for col in df.select_dtypes(include='number').columns:
        df[col].fillna(df[col].median(), inplace=True)
    st.success("Missing values handled.")

    # g) Pie charts
    st.subheader("g) Pie Charts")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Payment Type")
        st.pyplot(plt.pie(df["payment_type"].value_counts(), labels=df["payment_type"].value_counts().index, autopct='%1.1f%%')[0].figure)
    with col2:
        st.write("Trip Type")
        st.pyplot(plt.pie(df["trip_type"].value_counts(), labels=df["trip_type"].value_counts().index, autopct='%1.1f%%')[0].figure)

    # h–k) Groupby analysis
    st.subheader("h–k) GroupBy Analysis")
    st.write("Average Total Amount by Weekday:")
    st.write(df.groupby("weekday")["total_amount"].mean())

    st.write("Average Total Amount by Payment Type:")
    st.write(df.groupby("payment_type")["total_amount"].mean())

    st.write("Average Tip Amount by Weekday:")
    st.write(df.groupby("weekday")["tip_amount"].mean())

    st.write("Average Tip Amount by Payment Type:")
    st.write(df.groupby("payment_type")["tip_amount"].mean())

    # l–n) Hypothesis Testing
    st.subheader("l–n) Hypothesis Testing")

    l_stat, l_p = stats.f_oneway(*[df[df["trip_type"] == x]["total_amount"] for x in df["trip_type"].unique()])
    st.write(f"l) ANOVA for total_amount ~ trip_type: p-value = {l_p:.4f}")

    m_stat, m_p = stats.f_oneway(*[df[df["weekday"] == x]["total_amount"] for x in df["weekday"].unique()])
    st.write(f"m) ANOVA for total_amount ~ weekday: p-value = {m_p:.4f}")

    n_table = pd.crosstab(df["trip_type"], df["payment_type"])
    chi2, p_chi, _, _ = stats.chi2_contingency(n_table)
    st.write(f"n) Chi-square test trip_type vs payment_type: p-value = {p_chi:.4f}")

    # o) Numeric & p) Object columns
    numeric_vars = ["trip_distance", "fare_amount", "extra", "mta_tax", "tip_amount", "tolls_amount",
                    "improvement_surcharge", "congestion_surcharge", "trip_duration", "passenger_count"]
    object_vars = ["store_and_fwd_flag", "RatecodeID", "payment_type", "trip_type", "weekday", "hourofday"]

    # q) Correlation heatmap
    st.subheader("q) Correlation Analysis")
    corr = df[numeric_vars].corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # r) Dummy encoding
    df_encoded = pd.get_dummies(df[object_vars], drop_first=True)
    df_model = pd.concat([df[numeric_vars], df_encoded, df["total_amount"]], axis=1)

    # s) Target Variable Distribution
    st.subheader("s) Target Variable: total_amount")
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    sns.histplot(df["total_amount"], kde=True, ax=axs[0])
    axs[0].set_title("Histogram")
    sns.boxplot(x=df["total_amount"], ax=axs[1])
    axs[1].set_title("Boxplot")
    sns.kdeplot(df["total_amount"], ax=axs[2])
    axs[2].set_title("Density Plot")
    st.pyplot(fig)

    # t) Regression Models
    st.subheader("t) Regression Models")
    X = df_model.drop("total_amount", axis=1)
    y = df_model["total_amount"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=6),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        st.write(f"{name} R² Score: {r2:.3f}")

else:
    st.info("👆 Please upload a NYC Green Taxi Parquet file to start.")
