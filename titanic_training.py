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


# Plot: Age distribution by gender


# Plot scatter of age and fair, and overlay class as colourway




# Create logistic regression (gradient ascent algorithm) of survived
# 1: whole pop. have age () and sex (categorical) as dependent variables 




