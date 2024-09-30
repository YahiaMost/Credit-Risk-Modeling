# Credit Ristk Modeling

Learning credit risk data analysis and model development through a [certified online course ](https://www.udemy.com/course/credit-risk-modeling-in-python/). Currently in progress of finishing the course (>50% done) in order to better understand credit risk analysis at major banks. Upon completing the course I will be awarded a certificate which I can share at a later date. 


## Expected Loss Description

For a house with a value of $500,000\
The bank will fund 80\% = $400,000 (Loan To Value, LTV)\
The borrower already paid $40,000

Exposure at Default (EAD) = \$360,000\
Probability of Default (PD) = 1 in 4 borrowers = 25\%\
Current sale price of house = \$342,000\
Loss Given Default (LGD) = 360-342/360 = 5%

Expected Loss (EL) = PD x LGD x EAD\
Expected Loss (EL) = 25% x 5%x $360,000\
Expected Loss (EL) = $4500
