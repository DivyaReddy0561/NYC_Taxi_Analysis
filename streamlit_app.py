import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        results[name] = r2
        trained_models[name] = model

    return results, trained_models

# ---------- STREAMLIT APP ----------

st.set_page_config(page_title="NYC Taxi Fare Prediction", layout="wide")
st.title("🚕 NYC Taxi Fare Prediction App")

# Define Features
features = [
    'passenger_count', 'trip_distance', 'PULocationID', 'DOLocationID',
    'RatecodeID', 'payment_type', 'fare_amount', 'extra', 'mta_tax',
    'tip_amount', 'tolls_amount', 'improvement_surcharge',
    'trip_type', 'weekday', 'hour'
]

# Create dummy training data (simulate 1000 rows)
np.random.seed(42)
dummy_data = pd.DataFrame({
    'passenger_count': np.random.randint(1, 6, 1000),
    'trip_distance': np.random.uniform(0.5, 10, 1000),
    'PULocationID': np.random.randint(1, 200, 1000),
    'DOLocationID': np.random.randint(1, 200, 1000),
    'RatecodeID': np.random.randint(1, 6, 1000),
    'payment_type': np.random.randint(1, 5, 1000),
    'fare_amount': np.random.uniform(5, 50, 1000),
    'extra': np.random.uniform(0, 5, 1000),
    'mta_tax': np.random.uniform(0, 0.5, 1000),
    'tip_amount': np.random.uniform(0, 10, 1000),
    'tolls_amount': np.random.uniform(0, 10, 1000),
    'improvement_surcharge': np.random.uniform(0, 1, 1000),
    'trip_type': np.random.randint(1, 3, 1000),
    'weekday': np.random.randint(0, 7, 1000),
    'hour': np.random.randint(0, 24, 1000),
})

# Simulate target variable
dummy_data['total_amount'] = (
    dummy_data['fare_amount'] + dummy_data['extra'] + dummy_data['mta_tax'] +
    dummy_data['tip_amount'] + dummy_data['tolls_amount'] + dummy_data['improvement_surcharge']
)

X = dummy_data[features]
y = dummy_data['total_amount']

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
results, trained_models = train_and_evaluate_models(X_train, X_test, y_train, y_test)

# ---------- MODEL SELECTION ----------
st.subheader("📈 Model Selection & Performance")
selected_model_name = st.selectbox("Select a model for prediction", list(trained_models.keys()))
selected_model = trained_models[selected_model_name]
st.markdown(f"**R² Score** for {selected_model_name}: `{results[selected_model_name]:.3f}`")

# ---------- USER INPUT ----------
st.subheader("🔢 Input Ride Details")

user_input = {}
for col in features:
    if col in ['passenger_count', 'PULocationID', 'DOLocationID', 'RatecodeID', 'payment_type', 'trip_type', 'weekday', 'hour']:
        val = st.number_input(f"{col}", min_value=0, value=int(X[col].median()), step=1)
    else:
        val = st.number_input(f"{col}", value=float(round(X[col].median(), 2)))
    user_input[col] = val

# ---------- PREDICT ----------
if st.button("Predict Total Fare"):
    input_df = pd.DataFrame([user_input])
    prediction = selected_model.predict(input_df)[0]
    st.success(f"🎯 Predicted Total Fare Amount: **${prediction:.2f}**")
