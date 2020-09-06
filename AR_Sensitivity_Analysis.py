# -*- coding: utf-8 -*-
"""Finding the best AR Model to forecast with"""

#%% Import Library and data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Load specific forecasting tools
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse
from statsmodels.tsa.ar_model import AR,ARResults
# Ignore harmless warnings
import warnings
warnings.filterwarnings("ignore")

#%% Start taking time
import time
start_t = time.time()

#%% Import data
"""The data contains the summarized electrical load of 8 different households,
recorded in a 15 min intervall. The aim is to predict the load of the next 24h."""
df = pd.read_csv('data_prepared.csv', sep=',', parse_dates=True, index_col=0, dayfirst=True)
df.index.freq = '15T'

# Change data type from float64 zu int32 to reduce memory size
mez = df["MEZ/MESZ"] # save CET timestamp in a different DataFrame
df = df.drop(["MEZ/MESZ"], axis=1)
df = df.astype(float)

#%% Find the best Data amount to predict with
datasize = 'n'
if datasize == 'y':
    rmse_limit = {}
    for i in range(4,786):
        # Setting the datalimit from 4 to 786 days
        df_limit = df.iloc[-(i*96):]
        # Train Test Split
        prog = 96 # 24 h forecasting (e.g. 12h would be 48)
        train = df_limit.iloc[:-prog]
        test = df_limit.iloc[-prog:]
        # Define model and maxlag
        model = AR(train['Hges'])
        ARfit = model.fit(maxlag=96)
        # Do forecasting
        start=len(train)
        end=len(train)+len(test)-1
        predictions = ARfit.predict(start=start, end=end, dynamic=False).rename('Hges Predictions')
        # Evaluate forecast with rmse
        rmse_test = rmse(test['Hges'], predictions)
        # Save rmse in a dictionary
        rmse_limit[i] = rmse_test
        print(f'Data Limit {i}: ', rmse_test)
    rmse_limit = pd.DataFrame(rmse_limit.items(), columns=['Datenmenge in Tage', 'RMSE'])
    rmse_limit = rmse_limit.set_index('Datenmenge in Tage')
    print(f'Beste Datenmenge für dieses Modell:', rmse_limit['RMSE'].idxmin(), 
          'Tage mit RMSE', rmse_limit['RMSE'].min()) # 77 Tage mit RMSE 329.788 Wh
    plt.figure()
    rmse_limit.plot()
else:
    pass

#%% 24h forecasting with optimized data volume
num_d = 77 # 77 days was the best data volume for the analyzed dataset
df = df.iloc[-(num_d*96):]
mez = mez.iloc[-(num_d*96):]

# Train-Test Split
prog = 96 # 24 h forecasting
train = df.iloc[:-prog]
test = df.iloc[-prog:]

# Create AR(96)-model
model = AR(train['Hges'])
ARfit = model.fit(maxlag=96)

# Do forecasting
start=len(train)
end=len(train)+len(test)-1 # 1 weniger als len(df)
predictions = ARfit.predict(start=start, end=end, dynamic=False).rename('Hges Predictions')

# Compare forecast to real data in a plot
plt.figure()
test['Hges'].plot(legend=True)
predictions.plot(legend=True)

# Evaulation
print(f"Dateneinschränkung: {num_d} letzte Tage betrachtet")
from custom_fun import calculate_stats
title = "AR(" + str(prog) + ")"
calculate_stats(title, test['Hges'], predictions, len(train), len(ARfit.params))

#%% Stop taking time and measure
end_t = time.time()
diff = end_t - start_t
print(f'Berechnungsdauer: {diff:.2f} sec')
