import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
from scipy.stats import f_oneway, chi2_contingency

# Streamlit page config
st.set_page_config(page_title="NYC Taxi Analysis", layout="wide")
st.title("NYC Green Taxi Trip Data Analysis & ML Modeling")

# Upload file
uploaded_file = st.file_uploader("Upload a dataset (Parquet or CSV)", type=["parquet", "csv"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".parquet"):
        df = pd.read_parquet(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    st.subheader("Raw Data Sample")
    st.dataframe(df.head())

    # Drop unnecessary columns
    if "ehail_fee" in df.columns:
        df.drop("ehail_fee", axis=1, inplace=True)

    # Feature Engineering
    df["lpep_pickup_datetime"] = pd.to_datetime(df["lpep_pickup_datetime"])
    df["lpep_dropoff_datetime"] = pd.to_datetime(df["lpep_dropoff_datetime"])
    df["trip_duration"] = (df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]).dt.total_seconds() / 60
    df["weekday"] = df["lpep_pickup_datetime"].dt.day_name()
    df["hour"] = df["lpep_pickup_datetime"].dt.hour
    df = df[df["trip_duration"].between(1, 120)]

    # Handle missing values
    cat_cols = ['store_and_fwd_flag', 'RatecodeID', 'payment_type', 'trip_type']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    num_cols = df.select_dtypes(include=np.number).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    # Encode Categorical
    encoder_cols = ['store_and_fwd_flag', 'RatecodeID', 'payment_type', 'trip_type', 'PULocationID', 'DOLocationID']
    for col in encoder_cols:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    st.subheader("Processed Data Sample")
    st.dataframe(df.head())

    # Visualizations (g)
    st.subheader("Pie Charts")
    if 'payment_type' in df.columns:
        fig1, ax1 = plt.subplots()
        df['payment_type'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax1)
        ax1.set_ylabel("")
        ax1.set_title("Payment Type Distribution")
        st.pyplot(fig1)

    if 'trip_type' in df.columns:
        fig2, ax2 = plt.subplots()
        df['trip_type'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax2)
        ax2.set_ylabel("")
        ax2.set_title("Trip Type Distribution")
        st.pyplot(fig2)

    # Groupby analysis (h–k)
    st.subheader("Groupby Analysis")

    st.write("Average Total Amount by Weekday")
    st.dataframe(df.groupby("weekday")["total_amount"].mean().reset_index())

    st.write("Average Total Amount by Payment Type")
    st.dataframe(df.groupby("payment_type")["total_amount"].mean().reset_index())

    st.write("Average Tip Amount by Weekday")
    st.dataframe(df.groupby("weekday")["tip_amount"].mean().reset_index())

    st.write("Average Tip Amount by Payment Type")
    st.dataframe(df.groupby("payment_type")["tip_amount"].mean().reset_index())

    # Hypothesis Testing (l–n)
    st.subheader("Hypothesis Testing")

    # l) ANOVA - average total_amount by trip_type
    try:
        anova1 = f_oneway(*[group["total_amount"].values for name, group in df.groupby("trip_type")])
        st.write(f"ANOVA (total_amount ~ trip_type) p-value: {anova1.pvalue:.4f}")
    except:
        st.warning("ANOVA test failed for trip_type")

    # m) ANOVA - average total_amount by weekday
    try:
        anova2 = f_oneway(*[group["total_amount"].values for name, group in df.groupby("weekday")])
        st.write(f"ANOVA (total_amount ~ weekday) p-value: {anova2.pvalue:.4f}")
    except:
        st.warning("ANOVA test failed for weekday")

    # n) Chi-square test - trip_type vs payment_type
    if "trip_type" in df.columns and "payment_type" in df.columns:
        chi2_data = pd.crosstab(df['trip_type'], df['payment_type'])
        chi2_stat, p, dof, _ = chi2_contingency(chi2_data)
        st.write(f"Chi-square Test (trip_type vs payment_type) p-value: {p:.4f}")

    # o) Numeric variables
    st.subheader("Correlation Heatmap of Numeric Variables")
    numeric_vars = ['trip_distance', 'fare_amount', 'extra', 'mta_tax', 'tip_amount',
                    'tolls_amount', 'improvement_surcharge', 'trip_duration', 'passenger_count']
    if 'congestion_surcharge' in df.columns:
        numeric_vars.append('congestion_surcharge')
    
    corr = df[numeric_vars].corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # p) Object variables
    st.subheader("Object Variables")
    st.write(['store_and_fwd_flag', 'RatecodeID', 'payment_type', 'trip_type', 'weekday', 'hour'])

    # q) Dummy encode
    df_encoded = pd.get_dummies(df, columns=['store_and_fwd_flag', 'RatecodeID', 'payment_type', 'trip_type', 'weekday'], drop_first=True)

    # r) Histogram, Boxplot, Density for total_amount
    st.subheader("Target Variable: total_amount")

    fig, axs = plt.subplots(1, 3, figsize=(18, 4))
    sns.histplot(df['total_amount'], bins=40, ax=axs[0])
    axs[0].set_title("Histogram")

    sns.boxplot(x=df['total_amount'], ax=axs[1])
    axs[1].set_title("Boxplot")

    sns.kdeplot(df['total_amount'], ax=axs[2])
    axs[2].set_title("Density Plot")

    st.pyplot(fig)

    # s) Regression Models (final)
    st.subheader("Regression Models for Predicting total_amount")

    model_features = ['passenger_count', 'trip_distance', 'PULocationID', 'DOLocationID',
                      'fare_amount', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount',
                      'improvement_surcharge', 'trip_duration', 'hour']

    X = df_encoded[model_features]
    y = df_encoded['total_amount']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    models = {
        "Multiple Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=5, min_samples_leaf=10, random_state=42),
        "Random Forest (100 trees)": RandomForestRegressor(n_estimators=100, max_depth=6, min_samples_leaf=10, random_state=42),
        "Gradient Boosting (100 trees)": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        score = r2_score(y_test, preds)
        results[name] = score
        st.write(f"{name} - R² Score: **{score:.4f}**")

    st.subheader("Sample Predictions (Top 5 rows)")
    sample_df = X_test.iloc[:5].copy()
    sample_preds = {
        "Actual": y_test.iloc[:5].values
    }
    for name, model in models.items():
        sample_preds[name] = model.predict(sample_df)

    st.dataframe(pd.DataFrame(sample_preds))

else:
    st.info("📁 Please upload a Parquet or CSV file to begin.")
