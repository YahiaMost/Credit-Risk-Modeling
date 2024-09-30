# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 22:03:11 2024

@author: Yahia
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
import pprint as pprint

#%% Introduction
# =============================================================================
#                               Expected Loss Description
# =============================================================================

# For a house with a value of \\$500,000\
# The bank will fund 80\% = \\$400,000 (Loan To Value, LTV)\
# The borrower already paid \\$40,000\

# Exposure at Default (EAD) = \\$360,000\
# Probability of Default (PD) = 1 in 4 borrowers = 25\%\
# Current sale price of house = \\$342,000\
# Loss Given Default (LGD) = 360-342/360 = 5\%\

# Expected Loss (EL) = PD $\times$LGD$\times$EAD
# <center>Expected Loss (EL) = 25% $\times$5%$\times$\$360,000
# <center>Expected Loss (EL) = \$4500

loan_data_backup = pd.read_csv("loan_data_2007_2014.csv")

loan_data = loan_data_backup.copy() # Make a copy of the original data
pd.options.display.max_columns = None # Shows all columns of the DF
loan_data # Show us the DF (.head() or .tail() or .columns.values())
loan_data.info()

#%% Preprocessing Continuous Variables
# =============================================================================
# Preprocessing Continuous Variables
# =============================================================================
loan_data['emp_length'].unique()
loan_data['term'].unique()
# Convert values of employment length to float
loan_data['emp_length_int'] = loan_data['emp_length'].str.replace('10+ years','10')
loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace('< 1 year', str(0))
loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace('n/a',str(0))
loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace('years','')
loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace(' year','')
type(loan_data['emp_length_int'][0])
loan_data['emp_length_int'].unique()

# Convert values of term length to float
loan_data['term_int'] = loan_data['term'].str.replace(" months", "")
# Check conversion results
loan_data['emp_length_int'] = pd.to_numeric(loan_data['emp_length_int'])
type(loan_data['emp_length_int'][0])

loan_data['term_int'] = pd.to_numeric(loan_data['term_int'])
type(loan_data['term_int'][0])

# Calculate time since earliest credit line issue
loan_data['earliest_cr_line_date'] = pd.to_datetime(loan_data['earliest_cr_line'], format='%b-%y')
pd.to_datetime("2017-12-01") - loan_data['earliest_cr_line_date']
loan_data["days_since_earliest_cr_line"] = round(pd.to_numeric((pd.to_datetime("2017-12-01") - loan_data['earliest_cr_line_date']) / np.timedelta64(1, "D")))
loan_data["days_since_earliest_cr_line"].describe()
loan_data.loc[:,["earliest_cr_line", "earliest_cr_line_date", "days_since_earliest_cr_line"]][loan_data["days_since_earliest_cr_line"]<0]
loan_data["days_since_earliest_cr_line"][loan_data["days_since_earliest_cr_line"]<0] = loan_data["days_since_earliest_cr_line"].max()

loan_data["days_since_earliest_cr_line"].describe()

# Do the same to the issue_d (issue date) variable
loan_data['issue_d'].sample() # The dates are stored as strings
loan_data['issue_d_date'] = pd.to_datetime(loan_data['issue_d'], format='%b-%y')
loan_data['issue_d_date'].sample()
loan_data["days_since_issue_d"] = round(pd.to_numeric((pd.to_datetime("2017-12-01") - loan_data['issue_d_date']) / np.timedelta64(1, "D")))
loan_data["days_since_issue_d"].describe()
loan_data.info()

#%%% Get Dummies
pd.get_dummies(loan_data['grade'], prefix = "grade", prefix_sep= ":")
loan_data_dummies = [pd.get_dummies(loan_data['grade'], prefix = "grade", prefix_sep= ":"),
                    pd.get_dummies(loan_data['sub_grade'], prefix = "sub_grade", prefix_sep= ":"),
                    pd.get_dummies(loan_data['home_ownership'], prefix = "home_ownership", prefix_sep= ":"),
                    pd.get_dummies(loan_data['verification_status'], prefix = "verification_status", prefix_sep= ":"),
                    pd.get_dummies(loan_data['loan_status'], prefix = "loan_status", prefix_sep= ":"),
                    pd.get_dummies(loan_data['purpose'], prefix = "purpose", prefix_sep= ":"),
                    pd.get_dummies(loan_data['addr_state'], prefix = "addr_state", prefix_sep= ":"),
                    pd.get_dummies(loan_data['initial_list_status'], prefix = "initial_list_status", prefix_sep= ":")
                    ]

loan_data_dummies = pd.concat(loan_data_dummies, axis = 1)
loan_data = pd.concat([loan_data, loan_data_dummies], axis = 1)

loan_data.columns.values

#%%% Check Missing Values

loan_data.isnull()

# Data that is needed (null rows to be removed)
# annual_inc : annual income
# delinq_2yrs : delinquency (borrower is late or overdue) in the last 2 years
# inq_last_6mnths : inquiries in the last 6 months
# open_acc : open accounts
# pub_rec : public records
# total_acc : total accounts
# acc_now_delinq : accounts now delinquent
# total_rev_hi_lim : total revolving limit
# emp_lenght_int : employment length
# mths_since_earliest_cr_line : Months since earliest credit line

loan_data['total_rev_hi_lim'].fillna(loan_data['funded_amnt'], inplace=True) # replace the revolving amount missing data to be the total funded amount (inplace of the available row)
loan_data['total_rev_hi_lim'].isnull().sum()

loan_data['annual_inc'].fillna(loan_data['annual_inc'].mean(), inplace=True)
loan_data['annual_inc'].isnull().sum()

loan_data['days_since_earliest_cr_line'].fillna(0, inplace=True)
loan_data['acc_now_delinq'].fillna(0, inplace=True)
loan_data['total_acc'].fillna(0, inplace=True)
loan_data['pub_rec'].fillna(0, inplace=True)
loan_data['open_acc'].fillna(0, inplace=True)
loan_data['inq_last_6mths'].fillna(0, inplace=True)
loan_data['delinq_2yrs'].fillna(0, inplace=True)
loan_data['emp_length_int'].fillna(0, inplace=True)

#%% PD Model
# =============================================================================
#                   PD Model
# =============================================================================

#%%% Data Preparation
# Converting the continuous variables into dummy variables to use in the PD model to increase the simplicity of the model
loan_data['loan_status'].value_counts()/loan_data['loan_status'].count() # Proportion of accounts by status

# Default definition 
## We need to define the regression coefficients for each variable
loan_data["good_bad"] = np.where(loan_data['loan_status'].isin(['Charged Off', 'Default', 'Late (31-120 days)',
                                                               'Does not meet the credit policy. Status:Charged Off']), # Things that define default
                                0, # Value if the condition is true (defaulted)
                                1) # Value if the condition is false (non-default)

#%%% Fine-Classing
# Turning continuous variables into bins (discrete and categorical)

# Weight of evidence (WoE) shows to what extent an independent variable would predict a dependent variable

#%%% Coarse-Classing
# Combining bins that have a similar WoE (same strength predictability)
# Lowering the number of dummy variables thus improved the model

#%%% Splitting Data

# Lesson 25
# To prevent overfitting we split our original data into 90-10 or 80-20 to train-test the model
# Overfitting includes in it random noise that allows it to miss the point of the underlying data

from sklearn.model_selection import train_test_split

loan_data_inputs_train, loan_data_inputs_test, loan_data_targets_train, loan_data_targets_test = train_test_split(loan_data.drop('good_bad', axis=1),  # Independent Variables
                                                                                                                  loan_data['good_bad'], # Dependent Variables (PD)
                                                                                                                  test_size = 0.2, # 80-20 split
                                                                                                                  random_state= 42) # shuffle data in the same way every time
print("Train inputs:", loan_data_inputs_train.shape)
print("Train Targets:", loan_data_targets_train.shape)

print("Test inputs:", loan_data_inputs_test.shape)
print("Test Targets:", loan_data_targets_test.shape)

print("Ratio =", loan_data_targets_test.shape[0]*100/loan_data_targets_train.shape[0] ,"%")