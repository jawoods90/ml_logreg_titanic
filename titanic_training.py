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


# List of columns in the console, and then look at totals within some variables
df.columns
df['embark_town'].value_counts()
df['sex'].value_counts()
df['pclass'].value_counts()
df['class'].value_counts()

# test to see if survive and alive are same thing
tester = df[df['alive']=='yes']

# calculate some quick survival rates by class and sex

rsurvival_class = 
[df['alive']=='yes'].sum()
