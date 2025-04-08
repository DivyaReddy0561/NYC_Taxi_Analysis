import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score

# Define the list of features to use
features = [
    'passenger_count', 'trip_distance', 'PULocationID', 'DOLocationID',
    'RatecodeID', 'payment_type', 'fare_amount', 'extra', 'mta_tax',
    'tip_amount', 'tolls_amount', 'improvement_surcharge',
    'total_amount', 'trip_type', 'weekday', 'hour'
]

# Check if all features are present in the DataFrame
if not set(features).issubset(df.columns):
    st.error("Missing required features in dataset")
else:
    # Separate features (X) and target (y)
    X = df[features]
    y = df['trip_duration']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Initialize models with tuned hyperparameters
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=5, min_samples_leaf=10, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=50, max_depth=6, min_samples_leaf=10, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=4, random_state=42)
    }

    # Display model performance
    st.subheader("📊 Regression Model R² Scores")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        st.write(f"**{name}** R² score: **{r2:.3f}**")
