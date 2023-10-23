# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] papermill={"duration": 0.036252, "end_time": "2021-08-13T07:16:36.548737", "exception": false, "start_time": "2021-08-13T07:16:36.512485", "status": "completed"}
# # Telecom Churn Prediction - Starter Notebook
#
# **Author:** Eeshan Gupta  
# eeshangpt@gmail.com

# %% [markdown]
# ## Table of Content
# 1. [Problem Statememt](#Problem-statement)
# 2. A
#     2. V
#         3. B
#     4. A
# 1. A
#     2. V
#         3. B
#     4. A

# %% [markdown] papermill={"duration": 0.034552, "end_time": "2021-08-13T07:16:36.690028", "exception": false, "start_time": "2021-08-13T07:16:36.655476", "status": "completed"}
# ## Problem statement
#
# In the telecom industry, customers are able to choose from multiple service providers and actively switch from one operator to another. In this highly competitive market, the telecommunications industry experiences an average of 15-25% annual churn rate. Given the fact that it costs 5-10 times more to acquire a new customer than to retain an existing one, customer retention has now become even more important than customer acquisition.
#
# For many incumbent operators, retaining high profitable customers is the number one business
# goal. To reduce customer churn, telecom companies need to predict which customers are at high risk of churn. In this project, you will analyze customer-level data of a leading telecom firm, build predictive models to identify customers at high risk of churn, and identify the main indicators of churn.
#
# In this competition, your goal is *to build a machine learning model that is able to predict churning customers based on the features provided for their usage.*
#
# **Customer behaviour during churn:**
#
# Customers usually do not decide to switch to another competitor instantly, but rather over a
# period of time (this is especially applicable to high-value customers). In churn prediction, we
# assume that there are three phases of customer lifecycle :
#
# 1. <u>The ‘good’ phase:</u> In this phase, the customer is happy with the service and behaves as usual.
#
# 2. <u>The ‘action’ phase:</u> The customer experience starts to sore in this phase, for e.g. he/she gets a compelling offer from a competitor, faces unjust charges, becomes unhappy with service quality etc. In this phase, the customer usually shows different behaviour than the ‘good’ months. It is crucial to identify high-churn-risk customers in this phase, since some corrective actions can be taken at this point (such as matching the competitor’s offer/improving the service quality etc.)
#
# 3. <u>The ‘churn’ phase:</u> In this phase, the customer is said to have churned. In this case, since you are working over a four-month window, the first two months are the ‘good’ phase, the third month is the ‘action’ phase, while the fourth month (September) is the ‘churn’ phase.

# %% [markdown] papermill={"duration": 0.034501, "end_time": "2021-08-13T07:16:36.760335", "exception": false, "start_time": "2021-08-13T07:16:36.725834", "status": "completed"}
# ## Loading datasets

# %% [markdown]
# Importing Libraries

# %%
import os
import re
import warnings
from os import getcwd
from os.path import join, isfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# %%
from sklearn.impute import SimpleImputer

# %%
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# %% papermill={"duration": 1.362112, "end_time": "2021-08-13T07:16:38.158342", "exception": false, "start_time": "2021-08-13T07:16:36.796230", "status": "completed"}
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# from sklearn.linear_model import LogisticRegression
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import confusion_matrix, precision_score, recall_score

# %% papermill={"duration": 1.362112, "end_time": "2021-08-13T07:16:38.158342", "exception": false, "start_time": "2021-08-13T07:16:36.796230", "status": "completed"}
warnings.filterwarnings('ignore')
sns.set_style('darkgrid')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)
# %matplotlib inline

# %% [markdown] papermill={"duration": 0.03468, "end_time": "2021-08-13T07:16:38.240579", "exception": false, "start_time": "2021-08-13T07:16:38.205899", "status": "completed"}
# Next, we load our datasets and the data dictionary file.
#
# The **train.csv** file contains both dependent and independent features, while the **test.csv** contains only the independent variables. 
#
# So, for model selection, I will create our own train/test dataset from the **train.csv** and use the model to predict the solution using the features in unseen test.csv data for submission.

# %% [raw] papermill={"duration": 0.044801, "end_time": "2021-08-13T07:16:38.320264", "exception": false, "start_time": "2021-08-13T07:16:38.275463", "status": "completed"}
# #COMMENT THIS SECTION INCASE RUNNING THIS NOTEBOOK LOCALLY
#
# #Checking the kaggle paths for the uploaded datasets
# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# %%
PRJ_DIR = getcwd()
DATA_DIR = join(PRJ_DIR, 'input')

# %% papermill={"duration": 2.591587, "end_time": "2021-08-13T07:16:40.948543", "exception": false, "start_time": "2021-08-13T07:16:38.356956", "status": "completed"}
data = pd.read_csv(join(DATA_DIR, "train.csv"))
unseen = pd.read_csv(join(DATA_DIR, "test.csv"))
sample = pd.read_csv(join(DATA_DIR, "sample.csv"))
data_dict = pd.read_csv(join(DATA_DIR, "data_dictionary.csv"))

print(data.shape)
print(unseen.shape)
print(sample.shape)
print(data_dict.shape)

# %%
data.head()

# %%
empty_columns = data.columns[(data.isna().sum(axis=0) / data.shape[0]) > .50].tolist()
empty_columns

# %% [markdown]
# Dropping columns with >50% missing values

# %%
usable_columns = [i for i in data.columns if i not in empty_columns]

# %%
data = data[usable_columns]
y = data.pop('churn_probability')
usable_columns = data.columns.tolist()
data.head()

# %%
unseen = unseen[usable_columns]
unseen.head()

# %%
pd.DataFrame([(i, data[i].dtype, data[i].isna().sum(), data[i].nunique()) 
              for i in data.columns],
             columns=['name', 'type', 'num_null', 'num_unique'])

# %%
pd.DataFrame([(i, unseen[i].dtype, unseen[i].isna().sum(), unseen[i].nunique())
              for i in unseen.columns],
             columns=['name', 'type', 'num_null', 'num_unique'])

# %%
single_value_columns = [i for i in usable_columns
                        if data[i].nunique() <= 1]
usable_columns = [i for i in usable_columns
                  if i not in single_value_columns]
data = data[usable_columns]
unseen = unseen[usable_columns]

# %%
date_columns = data.columns[data.columns.str.contains('date')].tolist()
for i in date_columns:
    data[i] = pd.to_datetime(data[i])
    unseen[i] = pd.to_datetime(unseen[i])

# %%
usable_columns = [i for i in usable_columns
                  if i not in date_columns]
data = data[usable_columns + date_columns]
unseen = unseen[usable_columns + date_columns]

# %%
data.describe()

# %% [markdown]
# Class imbalance

# %%
plt.figure()
y.value_counts().plot(kind='bar')
plt.ylabel('Count of users')
plt.title('Class distribution')
plt.show()

# %%
y.value_counts() * 100 / y.value_counts().sum()

# %% [markdown]
# There is a huge imbalance among the customers who have and haven't churned. ~90% have not churned out while only ~10% churned.

# %%
imputer = SimpleImputer(strategy='median')
data[usable_columns] = imputer.fit_transform(data[usable_columns])
unseen[usable_columns] = imputer.transform(unseen[usable_columns])

# %%
for i in date_columns:
    data[i].fillna(data[i].median(), inplace=True)

# %%
minmax_scaler = MinMaxScaler()

# %%
data[date_columns] = minmax_scaler.fit_transform(data[date_columns])
unseen[date_columns] = minmax_scaler.transform(unseen[date_columns])

# %%
scaler = StandardScaler()

# %%
data.iloc[:, 1:] = scaler.fit_transform(data.iloc[:, 1:])
unseen.iloc[:, 1:] = scaler.transform(unseen.iloc[:, 1:])

# %%
data.head()

# %%
scaled_check = []
temp_ = []
for i in data.columns:
    if 'id' not in i:
        std = data[i].std()
        a = ((data[i] > (3 * std)) | (data[i] < -(3 * std)))
        scaled_check.append([i, a.sum(), (a.sum() * 100 / data.shape[0])])
        temp_.append(a)
pd.DataFrame(data=scaled_check, columns=["col_name", "dp_removed", "%_dp_removed"])

# %%
temp_

# %%
df = pd.DataFrame(temp_)

# %% [markdown]
# Outlier analysis remaining. 

# %% [markdown]
# Model training can be started 
