import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
from scipy.stats import f_oneway, chi2_contingency

df = pd.read_parquet("green_tripdata_2020-01.parquet")

df.drop("ehail_fee", axis=1, inplace=True, errors='ignore')
df["lpep_pickup_datetime"] = pd.to_datetime(df["lpep_pickup_datetime"])
df["lpep_dropoff_datetime"] = pd.to_datetime(df["lpep_dropoff_datetime"])
df["trip_duration"] = (df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]).dt.total_seconds() / 60
df["weekday"] = df["lpep_dropoff_datetime"].dt.dayofweek
df["hour"] = df["lpep_dropoff_datetime"].dt.hour
df = df[df["trip_duration"].between(1, 120)]

for col in ['store_and_fwd_flag', 'RatecodeID', 'payment_type', 'trip_type']:
    if col in df.columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

num_cols = df.select_dtypes(include=np.number).columns
for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

encode_cols = ['store_and_fwd_flag', 'RatecodeID', 'payment_type', 'trip_type', 'PULocationID', 'DOLocationID']
for col in encode_cols:
    if col in df.columns:
        df[col] = LabelEncoder().fit_transform(df[col])

df['payment_type'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title('Payment Type Distribution')
plt.show()

df['trip_type'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title('Trip Type Distribution')
plt.show()

df.groupby('weekday')['total_amount'].mean().plot(kind='bar', title='Average Total Amount by Weekday')
plt.show()

df.groupby('payment_type')['total_amount'].mean().plot(kind='bar', title='Average Total Amount by Payment Type')
plt.show()

df.groupby('weekday')['tip_amount'].mean().plot(kind='bar', title='Average Tip Amount by Weekday')
plt.show()

df.groupby('payment_type')['tip_amount'].mean().plot(kind='bar', title='Average Tip Amount by Payment Type')
plt.show()

trip_type_groups = [group["total_amount"] for name, group in df.groupby("trip_type")]
f_val, p_val = f_oneway(*trip_type_groups)
print(f"ANOVA - Total Amount by Trip Type: F-value = {f_val:.4f}, P-value = {p_val:.4f}")

weekday_groups = [group["total_amount"] for name, group in df.groupby("weekday")]
f_val2, p_val2 = f_oneway(*weekday_groups)
print(f"ANOVA - Total Amount by Weekday: F-value = {f_val2:.4f}, P-value = {p_val2:.4f}")

contingency = pd.crosstab(df["trip_type"], df["payment_type"])
chi2, p_chi, _, _ = chi2_contingency(contingency)
print(f"Chi-Square Test - Trip Type vs Payment Type: Chi2 = {chi2:.4f}, P-value = {p_chi:.4f}")

numeric_cols = ['trip_distance', 'fare_amount', 'extra', 'mta_tax', 'tip_amount',
                'tolls_amount', 'improvement_surcharge', 'trip_duration',
                'passenger_count', 'congestion_surcharge', 'total_amount']
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
sns.histplot(df["total_amount"], bins=40, ax=ax[0])
sns.boxplot(x=df["total_amount"], ax=ax[1])
sns.kdeplot(df["total_amount"], ax=ax[2])
ax[0].set_title("Histogram")
ax[1].set_title("Boxplot")
ax[2].set_title("KDE Curve")
plt.tight_layout()
plt.show()

features = ['passenger_count', 'trip_distance', 'PULocationID', 'DOLocationID',
            'RatecodeID', 'payment_type', 'fare_amount', 'extra', 'mta_tax',
            'tip_amount', 'tolls_amount', 'improvement_surcharge',
            'trip_type', 'weekday', 'hour']

X = df[features]
y = df["total_amount"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=5, min_samples_leaf=10, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=6, min_samples_leaf=10, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"{name}: R² Score = {r2:.4f}")
