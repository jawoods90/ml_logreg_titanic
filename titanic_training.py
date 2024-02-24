#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 17:26:45 2024

@author: josephwoods
"""

# import packages for use 

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Bring in dataset using seaborn
df = sns.load_dataset("titanic")
df = pd.DataFrame(df)


# List of columns in the console, and then look at totals within some variables
df.columns
df.dtypes
df['embark_town'].value_counts()
df['sex'].value_counts()
df['pclass'].value_counts()
df['class'].value_counts()
df['survived'].value_counts()


# create separate datasets for survivors, men, women and those in first class
df_survivors = df[df['alive']=='yes']
df_female= df[df['sex']=='female']
df_male = df[df['sex']=='male']
df_first = df[df['pclass']==1]


# calculate some quick survival rates by class and sex
n_all = len(df)
n_female = len(df_female)
n_male = len(df_male)

p_survive_all = len(df[df['survived']==1])/n_all
p_survive_male = len(df_male[df_male['survived']==1])/n_male
p_survive_female = len(df_female[df_female['survived']==1])/n_female


# Clean up data by dropping observations with no age
df_clean = df.dropna(subset=['age','fare'])


# Create Unique ID for each record, split out age and then merge back in
df_clean['unique_id'] = range(1, len(df_clean)+1)
df_age = df_clean[['unique_id', 'age']]
df_main = df_clean.drop(columns = ['age'])

df_clean2 = pd.merge(left = df_main, right = df_age, how = 'left', on = 'unique_id')


# Plot: Age distribution by gender
sns.kdeplot(data=df, x='age', hue='sex', multiple='stack')
sns.catplot(data=df, kind='swarm', x='class', y='age', hue='survived')

# Plot scatter of age and fair, and overlay class as colourway
sns.scatterplot(data=df_clean, x='age', y='fare', hue='class')

### Create logistic regression (gradient ascent algorithm) of survived
## 1: whole pop. have age (continuous) and sex (categorical) as dependent variables 

# MODEL 1: scikit learn log model version
# Import functions and split into training and test set
import sklearn as skl
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# create binary / boolean variable for gender
df_clean['sex_b'] = ''
df_clean['sex_b'] = np.where(df_clean['sex'] == 'female', True, False) 


features = ['age', 'sex_b'] # set of x variables
X = df_clean.loc[:, features] # get set of features
y = df_clean.loc[:, ['survived']] # get target variable

X_train, X_Test, y_train, y_test = train_test_split(X, y, train_size = 0.75, random_state = 42)

logreg = LogisticRegression(random_state=42)

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_Test)

# Get predictions from model 1 and merge onto y test
y_pred1 = pd.DataFrame(data=y_pred, columns=['y_pred']) 
y_test = pd.merge(left = y_test, right = y_pred1, how = 'left', left_index=True, right_index=False)

