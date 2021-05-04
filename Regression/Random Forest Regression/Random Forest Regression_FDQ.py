# Linear and Polynomial Regression
#
# This file is made by Faris D Qadri at 03.03.21
# Property of PCINU Germany e. V. @winter project 2020
# Github: https://github.com/NUJerman
# Source code: https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
#
# In this file I will explain you how to use linear and polynomial regression
# I will be using Bike_Sharing_hourly.csv as the dataframe
# 
# Contact person:
# Mail: abangfarisdq@gmail.com
# Linkedin: Faris Qadri
#
# Enjoy!!!
# NB: I only make the code more readable and easier to use. All credits goes to source

## Early visualisation at the data
# Pandas is used to see the data
import pandas as pd

# Data
print("==============================================================================\n")

print("Dataframe head:")
df = pd.read_csv("temperature.csv")
print(df.head())

# Statistical data insight
print("\ndf.describe result:")
print(df.describe())

print("==============================================================================\n")

# Use datetime for dealing with dates
import datetime

## making Timestamp readable
# Get years, months, and days
years = df["year"]
months = df["month"]
days = df["day"]

# List and then convert to datetime object
dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year,
         month, day in zip(years, months, days)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]

## Visualization using matplotlib
# Import matplotlib for visualization
import matplotlib.pyplot as plt

# %matplotlib inline # Please input this command if you are using Jupiter

# Set the style


# Set up the plotting layout
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize = (10,10))
fig.autofmt_xdate(rotation = 45)

# Actual max temperature measurement
ax1.plot(dates, df["actual"])
ax1.set_xlabel(""); ax1.set_ylabel("Temperature"); ax1.set_title("Max Temp")

# Temperature from 1 day ago
ax2.plot(dates, df["temp_1"])
ax2.set_xlabel(""); ax2.set_ylabel("Temperature"); ax2.set_title("Previous Max Temp")

# Temperature from 2 days ago
ax3.plot(dates, df["temp_2"])
ax3.set_xlabel("Date"); ax3.set_ylabel("Temperature"); ax3.set_title("Two Days Prior Max Temp")

# Friend Estimate
ax4.plot(dates, df["friend"])
ax4.set_xlabel("Date"); ax4.set_ylabel("Temperature"); ax4.set_title("Friend Estimate")


## This is where the fun begins
# One-hot encode categorical data
df = pd.get_dummies(df)
df.head(5)


# Use numpy to convert to arrays
import numpy as np

# Labels are the values we want to predict
labels = np.array(df["actual"])

# Remove the labels from the df
# axis 1 refers to the columns
df = df.drop("actual", axis = 1)

# Saving feature names for later use
df_list = list(df.columns)

# Convert to numpy array
df = np.array(df)

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
train_df, test_df, train_labels, test_labels = train_test_split(df, labels, test_size = 0.25,
                                                                           random_state = 42)

# Shape and size
print("Training Features Shape:", train_df.shape)
print("Training Labels Shape:", train_labels.shape)
print("Testing Features Shape:", test_df.shape)
print("Testing Labels Shape:", test_labels.shape)

print("\n==============================================================================\n")


# The baseline predictions are the historical averages
baseline_preds = test_df[:, df_list.index("average")]

# Baseline errors, and display average baseline error
baseline_errors = abs(baseline_preds - test_labels)
print("Average baseline error: ", round(np.mean(baseline_errors), 2),
      "degrees.")

print("\n==============================================================================\n")

## Train the model
# Import the model we are using
from sklearn.ensemble import RandomForestRegressor

# Instantiate model 
rf = RandomForestRegressor(n_estimators= 1000, random_state=42)

# Train the model on training data
rf.fit(train_df, train_labels);

## Making predictions on test data
# Use the forest's predict method on the test data
predictions = rf.predict(test_df)

# Calculate the absolute errors
errors = abs(predictions - test_labels)

# Print out the mean absolute error (mae)
print("Mean Absolute Error:", round(np.mean(errors), 2), "degrees.")

print("\n==============================================================================\n")

## Calculating accuracy of the model
# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)

# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print("Model accuracy:", round(accuracy, 2), "%.")

print("\n==============================================================================\n")



# Get numerical feature importances
importances = list(rf.feature_importances_)

# List of tuples with variable and importance
df_importances = [(feature, round(importance, 2)) for feature, importance in zip(df_list, importances)]

# Sort the feature importances by most important first
df_importances = sorted(df_importances, key = lambda x: x[1], reverse = True)

# Print out the feature and importances 
[print("Variable: {:20} Importance: {}".format(*pair)) for pair in df_importances];

print("\n==============================================================================\n")

## Predictions
# Dates of training values
months = df[:, df_list.index("month")]
days = df[:, df_list.index("day")]
years = df[:, df_list.index("year")]

# List and then convert to datetime object
dates = [str(int(year)) + "-" + str(int(month)) + "-" + str(int(day)) for year, month, day in zip(years, months, days)]
dates = [datetime.datetime.strptime(date, "%Y-%m-%d") for date in dates]

# Dataframe with true values and dates
true_data = pd.DataFrame(data = {"date": dates, "actual": labels})

# Dates of predictions
months = test_df[:, df_list.index("month")]
days = test_df[:, df_list.index("day")]
years = test_df[:, df_list.index("year")]

# Column of dates
test_dates = [str(int(year)) + "-" + str(int(month)) + "-" + str(int(day)) for year, month, day in zip(years, months, days)]

# Convert to datetime objects
test_dates = [datetime.datetime.strptime(date, "%Y-%m-%d") for date in test_dates]

# Dataframe with predictions and dates
predictions_data = pd.DataFrame(data = {"date": test_dates, "prediction": predictions})

## Plotting actual values
# Plot the actual values
plt.plot(true_data["date"], true_data["actual"], "b-", label = "actual")

# Plot the predicted values
plt.plot(predictions_data["date"], predictions_data["prediction"], "ro", label = "prediction")
plt.xticks(rotation = "60"); 
plt.legend()

