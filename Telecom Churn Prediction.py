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
# **Authors:** Eeshan Gupta  | Ashish Parmar <br/>
# eeshangpt@gmail.com | parmara266@gmail.com

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
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

# %% papermill={"duration": 1.362112, "end_time": "2021-08-13T07:16:38.158342", "exception": false, "start_time": "2021-08-13T07:16:36.796230", "status": "completed"}
np.random.seed(0)
warnings.filterwarnings('ignore')
sns.set_style('darkgrid')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)

# %matplotlib inline

# %%
PRJ_DIR = getcwd()
DATA_DIR = join(PRJ_DIR, 'input')

# %% [markdown] papermill={"duration": 0.03468, "end_time": "2021-08-13T07:16:38.240579", "exception": false, "start_time": "2021-08-13T07:16:38.205899", "status": "completed"}
# Next, we load our datasets and the data dictionary file.
#
# The **train.csv** file contains both dependent and independent features, while the **test.csv** contains only the independent variables. 
#
# So, for model selection, I will create our own train/test dataset from the **train.csv** and use the model to predict the solution using the features in unseen test.csv data for submission.

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

# %% [markdown]
# ### Missing Value Analysis

# %% [markdown]
# Finding columns with more than 50% of values as missing

# %%
empty_columns = data.columns[(data.isna().sum(axis=0) / data.shape[0]) > .50].tolist()
empty_columns

# %% [markdown]
# Dropping columns with more than 50% missing values

# %%
usable_columns = [i for i in data.columns if i not in empty_columns]
del empty_columns

# %%
data = data[usable_columns]
y = data.pop('churn_probability')
usable_columns = data.columns.tolist()
data.head()

# %%
unseen = unseen[usable_columns]
unseen.head()

# %% [markdown]
# Missing Value Analysis for training data

# %%
pd.DataFrame([(i, data[i].dtype, data[i].isna().sum(), data[i].nunique()) 
              for i in data.columns],
             columns=['name', 'type', 'num_null', 'num_unique'])

# %% [markdown]
# Missing Value Analysis for testing data

# %%
pd.DataFrame([(i, unseen[i].dtype, unseen[i].isna().sum(), unseen[i].nunique())
              for i in unseen.columns],
             columns=['name', 'type', 'num_null', 'num_unique'])

# %% [markdown]
# Dropping columns with only 1 value

# %%
single_value_columns = [i for i in usable_columns
                        if data[i].nunique() <= 1]
usable_columns = [i for i in usable_columns
                  if i not in single_value_columns]
data = data[usable_columns]
unseen = unseen[usable_columns]
del single_value_columns

# %% [markdown]
# Cleaning date columns

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

# %% [raw]
# ### ASDAASD

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

# %% [markdown]
# #### Imputing missing values

# %%
imputer = SimpleImputer(strategy='median')
data[usable_columns] = imputer.fit_transform(data[usable_columns])
unseen[usable_columns] = imputer.transform(unseen[usable_columns])

# %%
for i in date_columns:
    data[i].fillna(data[i].median(), inplace=True)

# %% [markdown]
# Scaling the values.

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

# %% [markdown]
# ### Outlier analysis
#
# Calculating the (+/-) 3 standard deviations for outlier ananlysis

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

temp_ = np.array([i.values for i in temp_])
a = temp_.sum(axis=0)
sns.histplot(a, kde=True, bins=10)

# %% [markdown]
# Removing those data points where they are outlier for 10 or more variables.

# %%
data = data[pd.Series(a) < 10]
y = y[pd.Series(a) < 10]

# %%
data.head()

# %% [markdown]
# Next piece of code is implements GridSearchCV for different models for different parameters. The code is commented since it takes a lot of time.  

# %%
# models = {
#     4: [
#         SVC(),
#         {
#             'kernel' : ['linear', 'poly', 'rbf', 'sigmoid'],
#             'degree': [3, 5, 7, 9, 10],
#             'gamma': ['scale', 'auto'],
#             'probability': [True],
#             'class_weight': ['balanced']
#         }
#     ],
#     3 : [
#         GradientBoostingClassifier(),
#         {
#             "loss": ['log_loss', 'exponential'],
#             "learning_rate": list(set(np.linspace(0, 1, 6).tolist() + np.linspace(0, 10, 6).tolist())),
#             "n_estimators": [100, 200, 300, 500],
#         }
#     ],
#     2: [
#         LogisticRegression(),
#         {
#             'penalty': ['l1', 'l2', None],
#             'class_weight': ['balanced', None],
#         }
#     ],
#     1: [
#         RandomForestClassifier(), dict()
#         {
#             "n_estimators": [100, 200, 500],
#             "criterion": ['gini'],
#             "min_samples_split": [5, 10],
#             "min_samples_leaf": [5, 9],
#             "bootstrap": [True],
#             "oob_score": [f1_score],
#             "class_weight": ["balanced", None],
#             "n_jobs": [2]
#         }
#     ], 
# }

# pca_model = PCA(n_components=0.95)
# x_trn, x_val, y_trn, y_val = train_test_split(data, y)
# pca_x_trn = pca_model.fit_transform(x_trn.iloc[:, 1:])
# pca_x_val = pca_model.transform(x_val.iloc[:, 1:])

# for model_id, model_details in models.items():
#     clf = GridSearchCV(estimator=model_details[0],
#                        param_grid=model_details[1],
#                        scoring='f1',
#                        n_jobs=4, verbose=True)
#     clf.fit(pca_x_trn, y_trn)
#     print('==========', model_id, sep='\n')
#     print(f"{clf.best_params_}", f"{clf.best_score_}", sep='\n')
#     print("==========")

# %%
models = [
    RandomForestClassifier(
        bootstrap=True, class_weight='balanced', criterion='gini',
        min_samples_leaf=9, min_samples_split=10,
        n_estimators=100, oob_score=f1_score
    )
]

# %%
X_trn, X_val, y_trn, y_val = train_test_split(data, y, train_size=0.8, random_state=0)

# %%
strat_k_folds = StratifiedKFold(n_splits=10)

# %%
# model_num, accuracy, f1_score, precision, recall
pca_n_com_s = np.linspace(0.75, 1., num=6)
pca_n_com = pca_n_com_s[-2]
overall_results = []
for i, model in enumerate(models):
    print(i)
    cross_validation_results = []
    for itr, (trn, val) in enumerate(strat_k_folds.split(data, y)):
        x_trn, y_trn = data.iloc[trn, 1:], y.iloc[trn]
        x_val, y_val = data.iloc[val, 1:], y.iloc[val]
        pca_model = PCA(n_components=pca_n_com)
        pca_x_trn = pca_model.fit_transform(x_trn.iloc[:, 1:])
        pca_x_val = pca_model.transform(x_val.iloc[:, 1:])
        model.fit(pca_x_trn, y_trn)
        pred = model.predict(pca_x_val)
        
        cross_validation_results.append([accuracy_score(y_val, pred),
                                         f1_score(y_val, pred),
                                         precision_score(y_val, pred),
                                         recall_score(y_val, pred)])
    cross_validation_results = np.array(cross_validation_results).mean(axis=0).tolist()
    overall_results.append([i] + cross_validation_results)
overall_results

# %% [raw]
# n_c = 0.95
#
# pca_model = PCA(n_components=n_c)
#
# pca_x_trn = pca_model.fit_transform(X_trn.iloc[:, 1:])
# pca_x_val = pca_model.transform(X_val.iloc[:, 1:])
#
# unseen_pca = pca_model.transform(unseen.iloc[:, 1:])
#
# pca_x_trn.shape, pca_x_val.shape, unseen_pca.shape
#
# model = SVC(class_weight='balanced')
#
# model.fit(pca_x_trn, y_trn)
#
# y_val_preds = model.predict(pca_x_val)
#
#
#
# accuracy_score(y_val, y_val_preds)
#
# f1_score(y_val, y_val_preds)
#
# y_tst_pred = model.predict(unseen_pca)
#
# y_unseen = sample['churn_probability']
#
# accuracy_score(y_unseen, y_tst_pred)
#
# f1_score(y_unseen, y_tst_pred)
#
# precision_score(y_unseen, y_tst_pred)
#
# recall_score(y_unseen, y_tst_pred)
#
# tn, fp, fn, tp =  confusion_matrix(y_unseen, y_tst_pred).ravel()
#
# tn
#
# tp
#
# fn
#
# fp

# %%
pd.DataFrame(overall_results, columns=["model_num",
                                       "accuracy", 
                                       "f1_score", 
                                       "precision", 
                                       "recall"])

# %%
