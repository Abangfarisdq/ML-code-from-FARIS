# Linear and Polynomial Regression
#
# This file is made by Faris D Qadri at 09.03.21
# Made intended for (but unoploaded):
# PCINU Germany e. V. @winter project 2020
# Github: https://github.com/NUJerman
# Source code: https://www.statsmodels.org/devel/examples/notebooks/generated/ordinal_regression.html
#
# In this file I will explain you how to use Ordinal Regression
# I will be using student application data from https://stats.idre.ucla.edu/stat/data/ologit.dta as the dataframe
# 
# Contact person:
# Mail: abangfarisdq@gmail.com
# Linkedin: Faris Qadri
#
# Enjoy!!!
# NB: I only make the code more readable and easier to use. All credits goes to source

## Data
# Necessary libraries
import numpy as np
import pandas as pd
import scipy.stats as stats

from statsmodels import OrderedModel

# Import dataset
print("==============================================================================\n")
url = "https://stats.idre.ucla.edu/stat/data/ologit.dta"
data_student = pd.read_stata(url)
print("\n", data_student.head(5))

# Datatype
print("==============================================================================\n")
print("\n", data_student.dtypes) 
data_student['apply'].dtype 

## Model start
# Import model
mod_prob = OrderedModel(data_student['apply'],
                        data_student[['pared', 'public', 'gpa']],
                        distr='probit')
# Check
print("==============================================================================\n")
res_prob = mod_prob.fit(method='bfgs')
print("\n", res_prob.summary())

# Start counting
num_of_thresholds = 2
mod_prob.transform_threshold_params(res_prob.params[-num_of_thresholds:])

# Model fitting
mod_log = OrderedModel(data_student['apply'],
                        data_student[['pared', 'public', 'gpa']],
                        distr='logit')

# Summary
print("==============================================================================\n")
res_log = mod_log.fit(method='bfgs', disp=False)
print("\n", res_log.summary())

print("==============================================================================\n")
predicted = res_log.model.predict(res_log.params, exog=data_student[['pared', 'public', 'gpa']])
print(predicted)

print("==============================================================================\n")
pred_choice = predicted.argmax(1)
print('\nFraction of correct choice predictions:')
print((np.asarray(data_student['apply'].values.codes) == pred_choice).mean())
print("==============================================================================\n")