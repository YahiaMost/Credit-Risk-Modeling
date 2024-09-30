# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 22:03:11 2024

@author: Yahia
"""
#%% # Expected Loss
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

import numpy as np
import pandas as pd

loan_data_backup = pd.read_csv("loan_data_2007_2014.csv")

#%% General Preprocessing
# =============================================================================
# General Preprocessing
# =============================================================================
loan_data = loan_data_backup.copy() # Make a copy of the original data
pd.options.display.max_columns = None # Shows all columns of the DF
loan_data # Show us the DF (.head() or .tail() or .columns.values())