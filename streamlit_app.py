import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
import streamlit as st

# Add slight noise to simulate real-world variation
dummy_data['trip_duration'] = dummy_data['trip_duration'] + np.random.normal(loc=0, scale=5, size=len(dummy_data))

# Feature selection (assuming features are defined)
X = dummy_data[features]
y = dummy_data['trip_duration']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=5, min_samples_leaf=10, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=50, max_depth=6, min_samples_leaf=10, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=4, random_state=42)
}

trained_models = {}
st.subheader("📊 Regression Model R² Scores")

# Train and display scores
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    trained_models[name] = model
    st.write(f"{name} R² Score: {r2:.3f}")
