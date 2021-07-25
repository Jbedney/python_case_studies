
import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
movie_data_path = #file path
df = pd.read_csv(movie_data_path, index_col=0)

df.head()

#Create column for profitability

df['profitable'] = np.where(df['revenue'] > df['budget'],True,False)
regression_target = 'revenue'
classification_target = 'profitable'
df.where(df['profitable'] == 1).count()

#Remove inf and null values

df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0,how='any')
df.shape

#split genres, separate distinct valyes, list all genres as new boolean columns 

genres_list = df.genres.apply(lambda x: x.split(","))
genres = []
for i in genres_list:
    i = [j.strip() for j in i]
    for j in i:
        if j not in genres:
            genres.append(j)

for j in genres:
    df[j] = df['genres'].str.contains(j).astype(int)

df.shape

#Define variable types, display, check for skew

continuous_covariates = ['budget', 'popularity', 'runtime', 'vote_count', 'vote_average']
outcomes_and_continuous_covariates = continuous_covariates + [regression_target, classification_target]
plotting_variables = ['budget', 'popularity', regression_target]

axes = pd.plotting.scatter_matrix(df[plotting_variables], alpha=0.15,        color=(0,0,0), hist_kwds={"color":(0,0,0)}, facecolor=(1,0,0))
# show the plot.
plt.show()
# determine the skew.
df[outcomes_and_continuous_covariates].skew()

#Normalize skew

right_skewed = ['budget','popularity','runtime','vote_count','revenue']
for column in df.columns:
    if column in right_skewed:
        df[column] = np.log10(1+df[column])
df[outcomes_and_continuous_covariates].skew()

#Save clean data

df.to_csv('movies_clean.csv')

