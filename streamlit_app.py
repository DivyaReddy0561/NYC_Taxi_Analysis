import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from scipy.stats import f_oneway, chi2_contingency

st.title("NYC Taxi Data Analysis and Fare Prediction")

# Load Data
df = pd.read_csv("yellow_tripdata_2020-01.csv")

# a) Info
st.subheader("a) Dataset Overview")
st.write(df.head())

# b) Drop 'ehail_fee'
if "ehail_fee" in df.columns:
    df.drop(columns=["ehail_fee"], inplace=True)

# c) Trip duration in minutes
df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])
df["trip_duration"] = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60

# d) Extract weekday
df["weekday"] = df["tpep_dropoff_datetime"].dt.day_name()

# e) Extract hour
df["hourofday"] = df["tpep_dropoff_datetime"].dt.hour

# f) Missing value imputation
df.fillna(df.median(numeric_only=True), inplace=True)
df.fillna("Missing", inplace=True)

# g) Pie charts
st.subheader("g) Pie Charts")
col1, col2 = st.columns(2)
with col1:
    st.write("Payment Type")
    plt.figure()
    plt.pie(df["payment_type"].value_counts(), labels=df["payment_type"].value_counts().index, autopct='%1.1f%%')
    st.pyplot()

with col2:
    st.write("Trip Type")
    plt.figure()
    plt.pie(df["trip_type"].value_counts(), labels=df["trip_type"].value_counts().index, autopct='%1.1f%%')
    st.pyplot()

# h-k) GroupBy summaries
st.subheader("h) Avg Total Amount by Weekday")
st.write(df.groupby("weekday")["total_amount"].mean())

st.subheader("i) Avg Total Amount by Payment Type")
st.write(df.groupby("payment_type")["total_amount"].mean())

st.subheader("j) Avg Tip by Weekday")
st.write(df.groupby("weekday")["tip_amount"].mean())

st.subheader("k) Avg Tip by Payment Type")
st.write(df.groupby("payment_type")["tip_amount"].mean())

# l) ANOVA for trip_type
st.subheader("l) ANOVA: Total Amount by Trip Type")
groups = [g["total_amount"].values for _, g in df.groupby("trip_type")]
anova_result = f_oneway(*groups)
st.write(anova_result)

# m) ANOVA for weekday
st.subheader("m) ANOVA: Total Amount by Weekday")
groups = [g["total_amount"].values for _, g in df.groupby("weekday")]
st.write(f_oneway(*groups))

# n) Chi-square test
st.subheader("n) Chi-square: Trip Type vs Payment Type")
chi2 = pd.crosstab(df["trip_type"], df["payment_type"])
chi_stat, p, dof, expected = chi2_contingency(chi2)
st.write(f"Chi-square = {chi_stat}, p = {p}")

# o) Numeric variables
numeric_cols = ["trip_distance", "fare_amount", "extra", "mta_tax", "tip_amount",
                "tolls_amount", "improvement_surcharge", "congestion_surcharge",
                "trip_duration", "passenger_count"]
st.subheader("o) Numeric Columns Summary")
st.write(df[numeric_cols].describe())

# p) Object variables
object_cols = ["store_and_fwd_flag", "RatecodeID", "payment_type", "trip_type", "weekday", "hourofday"]
st.subheader("p) Object Columns")
st.write(object_cols)

# q) Correlation
st.subheader("q) Correlation Heatmap")
plt.figure(figsize=(12, 5))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
st.pyplot()

# r) Dummy encoding
df_encoded = pd.get_dummies(df[object_cols], drop_first=True)
df_final = pd.concat([df[numeric_cols], df_encoded, df["total_amount"]], axis=1)

# s) Distribution Plots
st.subheader("s) Distribution of Total Amount")
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(df["total_amount"], bins=50, ax=ax[0])
ax[0].set_title("Histogram")
sns.boxplot(x=df["total_amount"], ax=ax[1])
ax[1].set_title("Boxplot")
sns.kdeplot(df["total_amount"], ax=ax[2])
ax[2].set_title("Density Plot")
st.pyplot(fig)

# t) Regression Model Training
st.subheader("t) Regression Models and Final Prediction")
X = df_final.drop("total_amount", axis=1)
y = df_final["total_amount"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=5),
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    score = r2_score(y_test, pred)
    st.write(f"{name}: R² Score = {score:.4f}")

# Final Prediction
sample = X_test.iloc[0:1]
prediction = models["Gradient Boosting"].predict(sample)[0]
st.write("Sample Input Features:")
st.write(sample)
st.write(f"🎯 Final Predicted Total Amount: **${prediction:.2f}**")
