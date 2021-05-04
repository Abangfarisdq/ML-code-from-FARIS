# Linear and Polynomial Regression
#
# This code is made by Faris D Qadri at 2021-03-03
# Property of PCINU Germany e. V. @winter project 2020
# Github: https://github.com/NUJerman
# Source: https://realpython.com/linear-regression-in-python/
#
# In this code I will explain you how to use linear and polynomial regression
# I will be using Bike_Sharing_hourly.csv as the dataframe
# 
# Contact person:
# Mail: abangfarisdq@gmail.com
# Linkedin: Faris Qadri
#
# Enjoy!!!

# Libraries that we will use
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

## Data
df = pd.read_csv("Bike_Sharing_hourly.csv")
df["Datetime"] = pd.to_datetime(df["Datetime"])
# df = df.resample("M", on="Datetime").Count.sum()
# This command is used to changed the frequency of the data into Monthly
# But somehow it can not doable because it wil make the data having only 1 colums
# If you found the answer, PLEASE contact me  

print(df)

# This variable is for fitting the model
X = df["Datetime"]
Y = df["Count"]

## Linear Regression
# Changing time to acceptable list
X = np.array(df["Datetime"].values.tolist())
X_linear = X.reshape(-1, 1)

# Fitting the model
model_linear = LinearRegression().fit(X_linear, Y)
Y_pred = model_linear.predict(X_linear)
r_sq_linear = model_linear.score(X_linear, Y)

# Result Linear
print("\nResult for Linear Regression\n")
print("coefficient of determination:", r_sq_linear, "\n")
print('intercept:', model_linear.intercept_, "\n")
print('slope:', model_linear.coef_, "\n")
print("============================================\n")


## Polynomial Regression
# Changing time to acceptable list
X_poly = X.reshape(-1, 1)
X_poly = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X_poly)

# Fitting the model
model_poly = LinearRegression().fit(X_poly, Y)
r_sq_poly = model_poly.score(X_poly, Y)
intercept_poly, coefficients_poly = model_poly.intercept_, model_poly.coef_
Y_pred_poly = model_poly.predict(X_poly)

# Result Poly
print("Result for Polynomial Regression\n")
print("coefficient of determination:", r_sq_poly, "\n")
print("intercept:", intercept_poly, "\n")
print("coefficients:", coefficients_poly, "\n")
print("predicted response:", Y_pred_poly, "\n")

