import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, f_oneway, chi2_contingency
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="NYC Taxi Analysis", layout="wide")

# App title
st.title("NYC Green Taxi Trip Data Analysis and Prediction")

# File uploader
uploaded_file = st.file_uploader("Upload a Parquet file", type="parquet")

if uploaded_file is not None:
    try:
        data = pd.read_parquet(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    # Data Processing
    # Drop 'ehail_fee'
    try:
        data = data.drop(columns=['ehail_fee'])
    except KeyError:
        st.warning("Column 'ehail_fee' not found. Continuing without dropping.")

    # Calculate trip duration
    data['trip_duration'] = (data['lpep_dropoff_datetime'] - data['lpep_pickup_datetime']).dt.total_seconds() / 60

    # Extract weekday and hour
    data['weekday'] = data['lpep_dropoff_datetime'].dt.day_name()
    data['hourofday'] = data['lpep_dropoff_datetime'].dt.hour

    # Impute missing values
    for column in data.columns:
        if data[column].dtype == 'object':
            try:
                data[column].fillna(data[column].mode()[0], inplace=True)
            except KeyError:
                st.warning(f"Cannot fillna for {column} as mode is unavailable")
        else:
            data[column].fillna(data[column].mean(), inplace=True)

    # Encoding categorical variables
    for col in data.select_dtypes(include='object'):
        try:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
        except Exception as e:
            st.warning(f"Could not encode {col} due to error: {e}")

    # Data Overview
    st.header("Data Overview")
    st.dataframe(data.head())

    # -------------------------------------------------------------------------
    # Visualizations and Statistical Analysis
    # -------------------------------------------------------------------------

    st.header("Visualizations and Statistical Analysis")

    # Payment Type Pie Chart
    st.subheader("Payment Type Distribution")
    payment_type_counts = data['payment_type'].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(payment_type_counts, labels=payment_type_counts.index, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig1)

    # Average Total Amount by Weekday
    st.subheader("Average Total Amount by Weekday")
    avg_total_by_weekday = data.groupby('weekday')['total_amount'].mean()
    st.bar_chart(avg_total_by_weekday)

    # Average Total Amount by Payment Type
    st.subheader("Average Total Amount by Payment Type")
    avg_total_by_payment_type = data.groupby('payment_type')['total_amount'].mean()
    st.bar_chart(avg_total_by_payment_type)

    # Hypothesis Testing
    st.subheader("Hypothesis Testing")

    # T-test for Trip Type
    try:
        trip_type_1 = data[data['trip_type'] == 1]['total_amount']
        trip_type_2 = data[data['trip_type'] == 2]['total_amount']
        t_stat, p_value = ttest_ind(trip_type_1, trip_type_2, nan_policy='omit')
        st.write(f"T-test result for trip type: t={t_stat:.2f}, p={p_value:.3f}")
    except Exception as e:
        st.error(f"Error during T-test: {e}")

    # ANOVA for Weekdays
    try:
        weekdays_groups = [data[data['weekday'] == day]['total_amount'] for day in data['weekday'].unique()]
        f_stat, p_value_weekday = f_oneway(*weekdays_groups)
        st.write(f"ANOVA test result for weekdays: F={f_stat:.2f}, p={p_value_weekday:.3f}")
    except Exception as e:
        st.error(f"Error during ANOVA: {e}")

    # Chi-square test for Trip Type and Payment Type
    try:
        contingency_table = pd.crosstab(data['trip_type'], data['payment_type'])
        chi2_stat, p_val_chi2, _, _ = chi2_contingency(contingency_table)
        st.write(f"Chi-square test result: chi2={chi2_stat:.2f}, p={p_val_chi2:.3f}")
    except Exception as e:
        st.error(f"Error during Chi-square test: {e}")

    # Correlation Matrix
    st.subheader("Correlation Matrix")
    numeric_cols = ['trip_distance', 'fare_amount', 'extra', 'mta_tax', 'tip_amount',
                    'tolls_amount', 'improvement_surcharge', 'congestion_surcharge',
                    'trip_duration', 'passenger_count']
    try:
        correlation_matrix = data[numeric_cols].corr()
        fig, ax = plt.subplots()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error creating correlation matrix: {e}")

    # Histograms, Boxplots, Density Curves of Total Amount
    st.subheader("Total Amount Analysis")
    fig2, axes = plt.subplots(1, 3, figsize=(15, 5))
    sns.histplot(data['total_amount'], bins=30, kde=False, ax=axes[0])
    axes[0].set_title('Histogram of Total Amount')
    sns.boxplot(x=data['total_amount'], ax=axes[1])
    axes[1].set_title('Boxplot of Total Amount')
    sns.kdeplot(data['total_amount'], shade=True, ax=axes[2])
    axes[2].set_title('Density Curve of Total Amount')
    st.pyplot(fig2)

    # -------------------------------------------------------------------------
    # Regression Models
    # -------------------------------------------------------------------------

    st.header("Regression Models")

    # Data Preparation for Regression
    try:
        X = data.drop(columns=['total_amount', 'lpep_pickup_datetime', 'lpep_dropoff_datetime'], errors='ignore')
        y = data['total_amount']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    except Exception as e:
        st.error(f"Error preparing data for regression: {e}")
        st.stop()

    # Model Training and Evaluation
    def train_and_evaluate(model, model_name):
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            st.write(f"{model_name} R^2:", r2)
            return model
        except Exception as e:
            st.error(f"Error training {model_name}: {e}")
            return None

    linear_model = train_and_evaluate(LinearRegression(), "Linear Regression")
    tree_model = train_and_evaluate(DecisionTreeRegressor(random_state=42), "Decision Tree")
    forest_model = train_and_evaluate(RandomForestRegressor(n_estimators=100, random_state=42), "Random Forest")
    gb_model = train_and_evaluate(GradientBoostingRegressor(n_estimators=100, random_state=42), "Gradient Boosting")

    # Prediction Section
    st.header("Make Predictions")
    st.write("Enter the feature values to predict the total amount.")

    input_data = {}
    for col in X.columns:
        if X[col].dtype == 'float64' or X[col].dtype == 'int64':
            input_data[col] = st.number_input(label=col, value=float(X[col].mean()))
        else:
            st.write(f"Skipping {col} as it is not numerical.")
    if st.button("Predict"):
    if linear_model is not None:
        input_df = pd.DataFrame([input_data])

        # Ensure input_df matches the training feature columns
        input_df = input_df[linear_model.feature_names_in_]

        prediction = linear_model.predict(input_df)[0]
        st.success(f"Predicted Total Amount: ${prediction:.2f}")
    else:
        st.error("Please train the Linear Regression model first.")




else:
    st.info("Upload a Parquet file to begin analysis.")
