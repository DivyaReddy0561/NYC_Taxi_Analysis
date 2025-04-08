import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score

st.set_page_config(page_title="NYC Taxi Fare Predictor", layout="wide")
st.title("🚕 NYC Taxi Fare Predictor")

# ------------------- Dataset Setup -------------------
features = [
    'passenger_count', 'trip_distance', 'PULocationID', 'DOLocationID',
    'RatecodeID', 'payment_type', 'fare_amount', 'extra', 'mta_tax',
    'tip_amount', 'tolls_amount', 'improvement_surcharge',
    'trip_type', 'weekday', 'hour'
]

np.random.seed(1)
df = pd.DataFrame({
    'passenger_count': np.random.randint(1, 6, 1000),
    'trip_distance': np.random.uniform(1, 15, 1000),
    'PULocationID': np.random.randint(1, 200, 1000),
    'DOLocationID': np.random.randint(1, 200, 1000),
    'RatecodeID': np.random.randint(1, 6, 1000),
    'payment_type': np.random.randint(1, 5, 1000),
    'fare_amount': np.random.uniform(5, 40, 1000),
    'extra': np.random.uniform(0, 5, 1000),
    'mta_tax': np.random.uniform(0, 0.5, 1000),
    'tip_amount': np.random.uniform(0, 10, 1000),
    'tolls_amount': np.random.uniform(0, 10, 1000),
    'improvement_surcharge': np.random.uniform(0, 1, 1000),
    'trip_type': np.random.randint(1, 3, 1000),
    'weekday': np.random.randint(0, 7, 1000),
    'hour': np.random.randint(0, 24, 1000),
})

# Target: Total amount = sum of various charges
df['total_amount'] = (
    df['fare_amount'] + df['extra'] + df['mta_tax'] +
    df['tip_amount'] + df['tolls_amount'] + df['improvement_surcharge']
)

X = df[features]
y = df['total_amount']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# ------------------- Model Training -------------------
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=5, min_samples_leaf=10, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=50, max_depth=6, min_samples_leaf=10, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=4, random_state=42)
}

r2_scores = {}
trained_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2_scores[name] = r2_score(y_test, y_pred)
    trained_models[name] = model

# ------------------- Display R2 Scores -------------------
st.subheader("📊 Model Performance (R² Scores)")
for model_name, r2 in r2_scores.items():
    st.markdown(f"**{model_name}**: R² = `{r2:.3f}`")

# ------------------- Select Model for Prediction -------------------
st.subheader("📌 Select a Model for Prediction")
selected_model_name = st.selectbox("Choose one model", list(trained_models.keys()))
selected_model = trained_models[selected_model_name]

# ------------------- Input Fields -------------------
st.subheader("📥 Input Trip Details")
user_input = {}
for col in features:
    if col in ['passenger_count', 'PULocationID', 'DOLocationID', 'RatecodeID', 'payment_type', 'trip_type', 'weekday', 'hour']:
        val = st.number_input(f"{col}", min_value=0, value=int(X[col].median()), step=1)
    else:
        val = st.number_input(f"{col}", value=float(round(X[col].median(), 2)))
    user_input[col] = val

# ------------------- Prediction -------------------
if st.button("🎯 Predict Total Amount"):
    input_df = pd.DataFrame([user_input])
    predicted_amount = selected_model.predict(input_df)[0]
    st.success(f"✅ Predicted Total Fare: **${predicted_amount:.2f}**")
