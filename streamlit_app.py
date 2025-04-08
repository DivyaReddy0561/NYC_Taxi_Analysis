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

# Page setup
st.set_page_config(page_title="NYC Green Taxi Analysis", layout="wide")
st.title("🗽 NYC Green Taxi Trip Analysis and Fare Prediction")

# Upload file
uploaded_file = st.file_uploader("📂 Upload NYC Green Taxi Trip Data (Parquet format)", type="parquet")

if uploaded_file:
    try:
        data = pd.read_parquet(uploaded_file)
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
        st.stop()

    # Preprocessing
    data.drop(columns=['ehail_fee'], errors='ignore', inplace=True)
    data['trip_duration'] = (data['lpep_dropoff_datetime'] - data['lpep_pickup_datetime']).dt.total_seconds() / 60
    data['weekday'] = data['lpep_dropoff_datetime'].dt.day_name()
    data['hourofday'] = data['lpep_dropoff_datetime'].dt.hour

    for col in data.columns:
        if data[col].dtype == 'object':
            try:
                data[col].fillna(data[col].mode()[0], inplace=True)
            except:
                pass
        else:
            data[col].fillna(data[col].mean(), inplace=True)

    # Label Encoding
    for col in data.select_dtypes(include='object'):
        try:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
        except Exception as e:
            st.warning(f"Encoding error in {col}: {e}")

    # ---------------- DATA OVERVIEW ----------------
    st.header("📊 Data Overview")
    st.dataframe(data.head())
    st.write("Shape:", data.shape)

    # ---------------- VISUALIZATIONS ----------------
    st.header("📈 Visual Insights")

    # Payment Type Pie
    try:
        st.subheader("🧾 Payment Type Distribution")
        payment_counts = data['payment_type'].value_counts()
        fig1, ax1 = plt.subplots()
        ax1.pie(payment_counts, labels=payment_counts.index, autopct='%1.1f%%', startangle=90)
        ax1.axis('equal')
        st.pyplot(fig1)
    except Exception as e:
        st.warning(f"Pie chart error: {e}")

    # Total Amount by Weekday
    st.subheader("📅 Average Total Amount by Weekday")
    st.bar_chart(data.groupby('weekday')['total_amount'].mean())

    # Total Amount by Payment Type
    st.subheader("💳 Average Total Amount by Payment Type")
    st.bar_chart(data.groupby('payment_type')['total_amount'].mean())

    # ---------------- STATISTICAL TESTS ----------------
    st.header("🧪 Hypothesis Testing")

    # T-Test
    try:
        t1 = data[data['trip_type'] == 1]['total_amount']
        t2 = data[data['trip_type'] == 2]['total_amount']
        t_stat, p_val = ttest_ind(t1, t2, nan_policy='omit')
        st.write(f"**T-Test (Trip Type):** t = {t_stat:.2f}, p = {p_val:.3f}")
    except Exception as e:
        st.warning(f"T-test error: {e}")

    # ANOVA
    try:
        groups = [data[data['weekday'] == day]['total_amount'] for day in data['weekday'].unique()]
        f_stat, p_val = f_oneway(*groups)
        st.write(f"**ANOVA (Weekdays):** F = {f_stat:.2f}, p = {p_val:.3f}")
    except Exception as e:
        st.warning(f"ANOVA error: {e}")

    # Chi-Square
    try:
        cont_table = pd.crosstab(data['trip_type'], data['payment_type'])
        chi2, p_chi2, _, _ = chi2_contingency(cont_table)
        st.write(f"**Chi-Square (Trip Type vs Payment):** χ² = {chi2:.2f}, p = {p_chi2:.3f}")
    except Exception as e:
        st.warning(f"Chi-square error: {e}")

    # ---------------- CORRELATION ----------------
    st.header("🔗 Correlation Heatmap")
    try:
        num_cols = ['trip_distance', 'fare_amount', 'extra', 'mta_tax', 'tip_amount',
                    'tolls_amount', 'improvement_surcharge', 'congestion_surcharge',
                    'trip_duration', 'passenger_count']
        corr = data[num_cols].corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Correlation matrix error: {e}")

    # ---------------- DISTRIBUTION ----------------
    st.header("📌 Total Fare Distribution")
    try:
        fig2, axes = plt.subplots(1, 3, figsize=(15, 5))
        sns.histplot(data['total_amount'], bins=30, ax=axes[0])
        axes[0].set_title("Histogram")
        sns.boxplot(x=data['total_amount'], ax=axes[1])
        axes[1].set_title("Boxplot")
        sns.kdeplot(data['total_amount'], shade=True, ax=axes[2])
        axes[2].set_title("Density Curve")
        st.pyplot(fig2)
    except Exception as e:
        st.warning(f"Distribution plots error: {e}")

    # ---------------- MODELING ----------------
    st.header("🤖 Fare Prediction with Regression Models")
    try:
        X = data.drop(columns=['total_amount', 'lpep_pickup_datetime', 'lpep_dropoff_datetime'], errors='ignore')
        y = data['total_amount']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    except Exception as e:
        st.error(f"Model data prep error: {e}")
        st.stop()

    def evaluate(model, name):
        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            score = r2_score(y_test, preds)
            st.write(f"**{name} R² Score:** {score:.3f}")
            return model
        except Exception as e:
            st.warning(f"{name} error: {e}")
            return None

    linear_model = evaluate(LinearRegression(), "Linear Regression")
    tree_model = evaluate(DecisionTreeRegressor(), "Decision Tree")
    rf_model = evaluate(RandomForestRegressor(), "Random Forest")
    gb_model = evaluate(GradientBoostingRegressor(), "Gradient Boosting")

    # ---------------- PREDICTION ----------------
    st.header("🔮 Predict Fare for Custom Input")
    st.write("Enter the input values for prediction:")

    input_values = {}
    for col in X.columns:
        if X[col].dtype in ['float64', 'int64']:
            input_values[col] = st.number_input(f"{col}", value=float(X[col].mean()))
        else:
            st.write(f"Skipping {col} (non-numeric)")

    if st.button("Predict Total Fare"):
        try:
            input_df = pd.DataFrame([input_values])
            for col in X.columns:
                if col not in input_df.columns:
                    input_df[col] = X[col].mean()
            input_df = input_df[X.columns]

            pred = linear_model.predict(input_df)[0]
            st.success(f"💰 Estimated Fare: ${pred:.2f}")
        except Exception as e:
            st.error(f"Prediction error: {e}")

else:
    st.info("📥 Please upload a `.parquet` file to proceed.")
