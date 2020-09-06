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

# Datentyp von float64 zu int32 ändern damit weniger Speicher benötigt wird
mez = df["MEZ/MESZ"] # MEZ separat speichern
df = df.drop(["MEZ/MESZ"], axis=1)
df = df.astype(float)

"""The datetime default format is: YYYY - MM - DD. 
For instance, January 2nd, 2019 
would look like: 2019-01-02.
"""

#%% Loop um Beste Datenmenge zu finden
datasize = 'n'
if datasize == 'y':
    rmse_limit = {}
    for i in range(4,786):
        # Datenlimit auf Tage
        df_limit = df.iloc[-(i*96):]
        # Train Test Split
        prog = 96 # 24 h prognostizieren
        train = df_limit.iloc[:-prog]
        test = df_limit.iloc[-prog:]
        # Modell definieren und fitten
        model = AR(train['Hges'])
        ARfit = model.fit(maxlag=96)
        # Prognose erstellen
        start=len(train)
        end=len(train)+len(test)-1 # 1 weniger als len(df)
        predictions = ARfit.predict(start=start, end=end, dynamic=False).rename('Hges Predictions')
        # Prognose bewerten
        rmse_test = rmse(test['Hges'], predictions)
        # An dic anhängen
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

#%% Prognose für die besagte Datenmenge plotten
# Daten einschränken: Geht nur wenn der Loop vorher lief, hier weiß ich allerdings dass 
# bei 77 Tagen die Beste Datenmenge liegt
num_d = 77 # Sind hier 77, der Wert ist veränderbar
df = df.iloc[-(num_d*96):]
mez = mez.iloc[-(num_d*96):]

# Train-Test Split
prog = 96 # 24 h prognostizieren
train = df.iloc[:-prog]
test = df.iloc[-prog:]

# AR(96)-Modell erstellen
# Modell definieren
model = AR(train['Hges'])
# Modell an den Trainingsdatensatz anpassen, Wichtige Sensitivitätsparameter:
# - maxlag beschreibt die Ordnung des Modells = 1, nutzt nur einen Lag Coefficient
# - method beschreibt die Methode mit der man auch probieren kann
# - solver
ARfit = model.fit(maxlag=96)

# Prognose erstellen
start=len(train)
end=len(train)+len(test)-1 # 1 weniger als len(df)
predictions = ARfit.predict(start=start, end=end, dynamic=False).rename('Hges Predictions')

# Vergleichsplot
# Damit eine neue figure erstellt wird und die folgenden nicht einfach in denselben Plot kommen
plt.figure()
test['Hges'].plot(legend=True)
predictions.plot(legend=True)

# Evaulation
print(f"Dateneinschränkung: {num_d} letzte Tage betrachtet")
from custom_fun import calculate_stats
title = "AR(" + str(prog) + ")"
calculate_stats(title, test['Hges'], predictions, len(train), len(ARfit.params))

#%% Zeitaufnahme beenden und Ergebnis ausgeben
end_t = time.time()
diff = end_t - start_t
print(f'Berechnungsdauer: {diff:.2f} sec')