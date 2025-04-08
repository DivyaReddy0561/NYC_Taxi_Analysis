import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score

# ---------- STREAMLIT APP ----------
st.set_page_config(page_title="NYC Trip Duration Prediction", layout="wide")
st.title("🚕 NYC Trip Duration Prediction App")

# Define Features
features = [
    'passenger_count', 'trip_distance', 'PULocationID', 'DOLocationID',
    'RatecodeID', 'payment_type', 'fare_amount', 'extra', 'mta_tax',
    'tip_amount', 'tolls_amount', 'improvement_surcharge',
    'total_amount', 'trip_type', 'weekday', 'hour'
]

# ---------- SIMULATE DATA ----------
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

# Create total_amount
dummy_data['total_amount'] = (
    dummy_data['fare_amount'] + dummy_data['extra'] + dummy_data['mta_tax'] +
    dummy_data['tip_amount'] + dummy_data['tolls_amount'] + dummy_data['improvement_surcharge']
)

# Simulate trip_duration (target variable)
dummy_data['trip_duration'] = dummy_data['trip_distance'] * np.random.uniform(3, 5, 1000) + np.random.normal(0, 2, 1000)

# ---------- MODELING ----------
X = dummy_data[features]
y = dummy_data['trip_duration']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=5, min_samples_leaf=10, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=50, max_depth=6, min_samples_leaf=10, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=4, random_state=42)
}

trained_models = {}
st.subheader("📊 Regression Model R² Scores")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    trained_models[name] = model
    st.write(f"**{name}** R² Score: {r2:.3f}")

# ---------- USER PREDICTION ----------
st.subheader("🛂 Enter Ride Details for Trip Duration Prediction")
user_input = {}
for col in features:
    if col in ['passenger_count', 'PULocationID', 'DOLocationID', 'RatecodeID', 'payment_type', 'trip_type', 'weekday', 'hour']:
        val = st.number_input(f"{col}", min_value=0, value=int(X[col].median()), step=1)
    else:
        val = st.number_input(f"{col}", value=float(round(X[col].median(), 2)))
    user_input[col] = val

# Model selection
selected_model_name = st.selectbox("Select a model to predict", list(trained_models.keys()))
selected_model = trained_models[selected_model_name]

# Prediction
if st.button("🎯 Predict Trip Duration"):
    input_df = pd.DataFrame([user_input])
    predicted_duration = selected_model.predict(input_df)[0]
    st.success(f"Predicted Trip Duration: **{predicted_duration:.2f} minutes**")
