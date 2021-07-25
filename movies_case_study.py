#!/usr/bin/env python
# coding: utf-8

# ### Exercise 1
# 
# First, we will import several libraries. `scikit-learn` (**sklearn**) contains helpful statistical models, and we'll use the `matplotlib.pyplot` library for visualizations. Of course, we will use `numpy` and `pandas` for data manipulation throughout.
# 
# #### Instructions 
# 
# - Read and execute the given code.
# - Call `df.head()` to take a look at the data.

# In[1]:


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

df = pd.read_csv("https://courses.edx.org/asset-v1:HarvardX+PH526x+2T2019+type@asset+block@movie_data.csv", index_col=0)

# Enter code here.
df.head()


# ### Exercise 2
# 
# In this exercise, we will define the regression and classification outcomes. Specifically, we will use the `revenue` column as the target for regression. For classification, we will construct an indicator of profitability for each movie.
# 
# #### Instructions 
# - Create a new column in `df` called `profitable`, defined as 1 if the movie `revenue` is greater than the movie `budget`, and 0 otherwise.
# - Next, define and store the outcomes we will use for regression and classification.
#     - Define `regression_target` as the string `'revenue'`.
#     - Define `classification_target` as the string `'profitable'`.

# In[2]:


df['profitable'] = np.where(df['revenue'] > df['budget'],True,False)
regression_target = 'revenue'
classification_target = 'profitable'
df.where(df['profitable'] == 1).count()


# ### Exercise 3
# 
# For simplicity, we will proceed by analyzing only the rows without any missing data. In this exercise, we will remove rows with any infinite or missing values.
# 
# #### Instructions 
# 
# - Use `df.replace()` to replace any cells with type `np.inf` or `-np.inf` with `np.nan`.
# - Drop all rows with any `np.nan` values in that row using `df.dropna()`. Do any further arguments need to be specified in this function to remove rows with any such values?

# In[3]:


df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0,how='any')
df.shape


# ### Exercise 4
# 
# Many of the variables in our dataframe contain the names of genre, actors/actresses, and keywords. Let's add indicator columns for each genre.
# 
# #### Instructions 
# 
# - Determine all the genres in the genre column. Make sure to use the `strip()` function on each genre to remove trailing characters.
# - Next, include each listed genre as a new column in the dataframe. Each element of these genre columns should be 1 if the movie belongs to that particular genre, and 0 otherwise. Keep in mind, a movie may belong to several genres at once.
# - Call `df[genres].head()` to view your results.

# In[4]:


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


# ### Exercise 5
# 
# Some variables in the dataset are already numeric and perhaps useful for regression and classification. In this exercise, we will store the names of these variables for future use. We will also take a look at some of the continuous variables and outcomes by plotting each pair in a scatter plot. Finally, we will evaluate the skew of each variable.
# 
# #### Instructions 
# - Call `plt.show()` to observe the plot below.
#     - Which of the covariates and/or outcomes are correlated with each other?
# - Call `skew()` on the columns `outcomes_and_continuous_covariates` in df.
#     - Is the skew above 1 for any of these variables?

# In[5]:


continuous_covariates = ['budget', 'popularity', 'runtime', 'vote_count', 'vote_average']
outcomes_and_continuous_covariates = continuous_covariates + [regression_target, classification_target]
plotting_variables = ['budget', 'popularity', regression_target]

axes = pd.plotting.scatter_matrix(df[plotting_variables], alpha=0.15,        color=(0,0,0), hist_kwds={"color":(0,0,0)}, facecolor=(1,0,0))
# show the plot.
plt.show()
# determine the skew.
df[outcomes_and_continuous_covariates].skew()


# ### Exercise 6
# 
# It appears that the variables `budget`, `popularity`, `runtime`, `vote_count`, and `revenue` are all right-skewed. In this exercise, we will transform these variables to eliminate this skewness. Specifically, we will use the `np.log10()` method. Because some of these variable values are exactly 0, we will add a small positive value to each to ensure it is defined; this is necessary because log(0) is negative infinity.
# 
# #### Instructions 
# - For each above-mentioned variable in `df`, transform value `x` into `np.log10(1+x)`.

# In[6]:


right_skewed = ['budget','popularity','runtime','vote_count','revenue']
for column in df.columns:
    if column in right_skewed:
        df[column] = np.log10(1+df[column])
df[outcomes_and_continuous_covariates].skew()


# ### Exercise 7
# 
# Let's now save our dataset. 
# 
# #### Instructions 
# - Use `to_csv()` to save the `df` object as `movies_clean.csv`.

# In[7]:


df.to_csv('movies_clean.csv')

