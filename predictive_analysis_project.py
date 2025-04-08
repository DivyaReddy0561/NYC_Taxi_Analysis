# -*- coding: utf-8 -*-
"""Predictive Analysis_Project"""

# Install necessary libraries
!pip install pyarrow matplotlib seaborn scikit-learn

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, chi2_contingency, f_oneway
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
data = pd.read_parquet("/content/green_tripdata_2022-01.parquet")

# a) Display info about the dataset
data.info()

# b) Drop the 'ehail_fee' column
data = data.drop(columns=['ehail_fee'])

# c) Calculate trip_duration in minutes
data['trip_duration'] = (data['lpep_dropoff_datetime'] - data['lpep_pickup_datetime']).dt.total_seconds() / 60

# d) Extract weekday and calculate value counts
data['weekday'] = data['lpep_dropoff_datetime'].dt.day_name()
print(data['weekday'].value_counts())

# e) Extract hour of day and calculate value counts
data['hourofday'] = data['lpep_dropoff_datetime'].dt.hour
print(data['hourofday'].value_counts())

# f) Impute missing values (mean for numeric, mode for object)
for column in data.columns:
    if data[column].dtype == 'object':
        data[column].fillna(data[column].mode()[0], inplace=True)
    else:
        data[column].fillna(data[column].mean(), inplace=True)

# g) Pie diagram of payment_type
payment_type_counts = data['payment_type'].value_counts()
plt.figure(figsize=(8, 8))
payment_type_counts.plot.pie(autopct='%1.1f%%', startangle=90, labels=payment_type_counts.index)
plt.title('Payment Type Distribution')
plt.ylabel('')
plt.show()

# h) Groupby average total_amount by weekday
avg_total_by_weekday = data.groupby('weekday')['total_amount'].mean()
print(avg_total_by_weekday)

# i) Groupby average total_amount by payment_type
avg_total_by_payment_type = data.groupby('payment_type')['total_amount'].mean()
print(avg_total_by_payment_type)

# j) T-test: average total_amount of different trip_type is identical
trip_type_1 = data[data['trip_type'] == 1]['total_amount']
trip_type_2 = data[data['trip_type'] == 2]['total_amount']
t_stat, p_value = ttest_ind(trip_type_1, trip_type_2, nan_policy='omit')
print("T-test result for trip type:", t_stat, p_value)

# k) ANOVA test for weekdays
weekdays_groups = [group["total_amount"].values for _, group in data.groupby("weekday")]
f_stat, p_value_weekday = f_oneway(*weekdays_groups)
print("ANOVA test result for weekdays:", f_stat, p_value_weekday)

# l) Chi-square test: trip_type vs payment_type
contingency_table = pd.crosstab(data['trip_type'], data['payment_type'])
chi2_stat, p_val_chi2, _, _ = chi2_contingency(contingency_table)
print("Chi-square test result:", chi2_stat, p_val_chi2)

# m) Correlation matrix
numeric_cols = ['trip_distance', 'fare_amount', 'extra', 'mta_tax', 'tip_amount',
                'tolls_amount', 'improvement_surcharge', 'congestion_surcharge',
                'trip_duration', 'passenger_count']
correlation_matrix = data[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# n) Dummy encode object variables
object_cols = ['store_and_fwd_flag', 'RatecodeID', 'payment_type',
               'trip_type', 'weekday', 'hourofday']
data_encoded = pd.get_dummies(data, columns=object_cols)

# o) Histogram, Boxplot, Density Curve of total_amount
plt.figure(figsize=(16, 6))
plt.subplot(1, 3, 1)
sns.histplot(data['total_amount'], bins=30, kde=False)
plt.title('Histogram of Total Amount')

plt.subplot(1, 3, 2)
sns.boxplot(x=data['total_amount'])
plt.title('Boxplot of Total Amount')

plt.subplot(1, 3, 3)
sns.kdeplot(data['total_amount'], shade=True)
plt.title('Density Curve of Total Amount')
plt.show()

# ✅ FINAL BLOCK: Define X and y, avoiding leakage
leaky_cols = [
    'total_amount',
    'fare_amount',
    'tip_amount',
    'tolls_amount',
    'extra',
    'mta_tax',
    'congestion_surcharge',
    'improvement_surcharge',
    'lpep_pickup_datetime',
    'lpep_dropoff_datetime'
]

# Drop features that directly sum into total_amount (to prevent data leakage)
leakage_features = ['total_amount', 'fare_amount', 'extra', 'mta_tax',
                    'tip_amount', 'tolls_amount', 'improvement_surcharge',
                    'congestion_surcharge', 'lpep_pickup_datetime', 'lpep_dropoff_datetime']

X = data_encoded.drop(columns=leakage_features)
y = data_encoded['total_amount']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train & Evaluate Regression Models

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)
print("Linear Regression R^2:", r2_score(y_test, y_pred_linear))

# Decision Tree Regressor
tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)
print("Decision Tree R^2:", r2_score(y_test, y_pred_tree))

# Random Forest Regressor
forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
forest_model.fit(X_train, y_train)
y_pred_forest = forest_model.predict(X_test)
print("Random Forest R^2:", r2_score(y_test, y_pred_forest))

# Gradient Boosting Regressor
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)
print("Gradient Boosting R^2:", r2_score(y_test, y_pred_gb))
