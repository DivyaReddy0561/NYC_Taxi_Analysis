NYC Green Taxi Trip Data Analysis & Fare Prediction
- Project Overview

This project presents an end-to-end data analysis and machine learning workflow on the NYC Green Taxi Trip dataset. The objective is to explore taxi trip patterns, analyze passenger and payment behaviors, validate insights using statistical methods, and build predictive models to estimate the total taxi fare.

The project combines exploratory data analysis (EDA), hypothesis testing, and regression modeling, and is deployed as an interactive Streamlit web application for real-time analysis and prediction.

- Dataset

NYC TLC Green Taxi Trip Data (January 2022)

Large-scale real-world transportation dataset in Parquet format

Contains trip timestamps, distances, fare details, payment types, trip types, and passenger counts

- Tools & Technologies

Python

Pandas, NumPy

Matplotlib, Seaborn

SciPy (Statistical Analysis)

Scikit-learn (Machine Learning)

Streamlit (Interactive Dashboard)

PyArrow (Parquet file handling)

- Data Preprocessing & Feature Engineering

Removed irrelevant and sparse columns

Handled missing values using mean and mode imputation

Created new features:

Trip duration (in minutes)

Day of week

Hour of day

Encoded categorical variables for modeling

- Exploratory Data Analysis (EDA)

Analyzed payment type distribution

Studied fare variation across weekdays and payment methods

Visualized fare distribution using histograms, boxplots, and density curves

Generated correlation heatmaps to understand relationships among numerical variables

- Statistical Hypothesis Testing

T-Test to compare average fare between different trip types

ANOVA to analyze fare variation across weekdays

Chi-Square Test to identify association between trip type and payment method

These tests ensured that observed patterns were statistically significant.

- Machine Learning Models

The project predicts the total trip fare (total_amount) using multiple regression models:

Linear Regression

Decision Tree Regressor

Random Forest Regressor

Gradient Boosting Regressor

Model performance was evaluated using the R² score, with tree-based ensemble models showing better performance due to non-linear relationships in the data.

- Interactive Streamlit Application

An interactive dashboard was built using Streamlit, allowing users to:

Upload taxi trip data

Explore visual insights and statistical results

Train machine learning models

Predict total taxi fare based on input features



This project demonstrates practical skills in:

Data analysis and visualization

Statistical reasoning
