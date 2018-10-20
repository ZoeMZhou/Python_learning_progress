# Task 1
print("that works")

# Task 2

import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import pylab
import statsmodels.formula.api as sm
import quandl



all_data={}
for ticker in ['AAL','ALK','WTI']:
all_data[ticker] = quandl.get("WIKI/{}".format(ticker), start_date="2014-06-01", end_date="2016-06-13", api_key='pU6qwHx4VSxzsPUjXjPG')
print(all_data['WTI'].head())
print(all_data['AAL'].head())
print(all_data['ALK'].head())