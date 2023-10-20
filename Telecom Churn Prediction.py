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
# **Author:** Akshay Sehgal (www.akshaysehgal.com)

# %% [markdown] papermill={"duration": 0.035836, "end_time": "2021-08-13T07:16:36.620553", "exception": false, "start_time": "2021-08-13T07:16:36.584717", "status": "completed"}
# The goal of this notebook is to provide an overview of how write a notebook and create a submission file that successfully solves the churn prediction problem. Please download the datasets, unzip and place them in the same folder as this notebook.
#
# We are going to follow the process called CRISP-DM.
#
# <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b9/CRISP-DM_Process_Diagram.png/639px-CRISP-DM_Process_Diagram.png" style="height: 400px; width:400px;"/>
#
# After Business and Data Understanding via EDA, we want to prepare data for modelling. Then evaluate and submit our predictions.

# %% [markdown] papermill={"duration": 0.034552, "end_time": "2021-08-13T07:16:36.690028", "exception": false, "start_time": "2021-08-13T07:16:36.655476", "status": "completed"}
# # 0. Problem statement
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
# # 1. Loading dependencies & datasets
#
# Lets start by loading our dependencies. We can keep adding any imports to this cell block, as we write mode and mode code.

# %% papermill={"duration": 1.362112, "end_time": "2021-08-13T07:16:38.158342", "exception": false, "start_time": "2021-08-13T07:16:36.796230", "status": "completed"}
#Data Structures
import pandas as pd
import numpy as np
import re
import os

### For installing missingno library, type this command in terminal
#pip install missingno

import missingno as msno

#Sklearn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, precision_score, recall_score

#Plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

#Others
import warnings
warnings.filterwarnings('ignore')

# %matplotlib inline

# %% [markdown] papermill={"duration": 0.03468, "end_time": "2021-08-13T07:16:38.240579", "exception": false, "start_time": "2021-08-13T07:16:38.205899", "status": "completed"}
# Next, we load our datasets and the data dictionary file.
#
# The **train.csv** file contains both dependent and independent features, while the **test.csv** contains only the independent variables. 
#
# So, for model selection, I will create our own train/test dataset from the **train.csv** and use the model to predict the solution using the features in unseen test.csv data for submission.

# %% papermill={"duration": 0.044801, "end_time": "2021-08-13T07:16:38.320264", "exception": false, "start_time": "2021-08-13T07:16:38.275463", "status": "completed"}
#COMMENT THIS SECTION INCASE RUNNING THIS NOTEBOOK LOCALLY

#Checking the kaggle paths for the uploaded datasets
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# %% papermill={"duration": 2.591587, "end_time": "2021-08-13T07:16:40.948543", "exception": false, "start_time": "2021-08-13T07:16:38.356956", "status": "completed"}
#INCASE RUNNING THIS LOCALLY, PASS THE RELATIVE PATH OF THE CSV FILES BELOW
#(e.g. if files are in same folder as notebook, simple write "train.csv" as path)

data = pd.read_csv("/kaggle/input/kagglelabstest2021/train.csv")
unseen = pd.read_csv("/kaggle/input/kagglelabstest2021/test.csv")
sample = pd.read_csv("/kaggle/input/kagglelabstest2021/sample.csv")
data_dict = pd.read_csv("/kaggle/input/kagglelabstest2021/data_dictionary.csv")

print(data.shape)
print(unseen.shape)
print(sample.shape)
print(data_dict.shape)

# %% [markdown] papermill={"duration": 0.035613, "end_time": "2021-08-13T07:16:41.020316", "exception": false, "start_time": "2021-08-13T07:16:40.984703", "status": "completed"}
# 1. Lets analyze the data dictionary versus the churn dataset.
# 2. The data dictonary contains a list of abbrevations which provide you all the information you need to understand what a specific feature/variable in the churn dataset represents
# 3. Example: 
#
# > "arpu_7" -> Average revenue per user + KPI for the month of July
# >
# > "onnet_mou_6" ->  All kind of calls within the same operator network + Minutes of usage voice calls + KPI for the month of June
# >
# >"night_pck_user_8" -> Scheme to use during specific night hours only + Prepaid service schemes called PACKS + KPI for the month of August
# >
# >"max_rech_data_7" -> Maximum + Recharge + Mobile internet + KPI for the month of July
#
# Its important to understand the definitions of each feature that you are working with, take notes on which feature you think might impact the churn rate of a user, and what sort of analysis could you do to understand the distribution of the feature better.

# %% papermill={"duration": 0.067353, "end_time": "2021-08-13T07:16:41.123346", "exception": false, "start_time": "2021-08-13T07:16:41.055993", "status": "completed"}
data_dict

# %% [markdown] papermill={"duration": 0.036844, "end_time": "2021-08-13T07:16:41.198447", "exception": false, "start_time": "2021-08-13T07:16:41.161603", "status": "completed"}
# For the purpose of this **starter notebook**, we I will restrict the dataset to only a small set of variables. 
#
# The approach I use here is to understand each Acronym, figure our what variable might be important and filter out variable names based on the combinations of acrynoms using REGEX. So, if I want the total minutes a person has spent on outgoing calls, I need acronyms, TOTAL, OG and MOU. So corresponding regex is ```total.+og.+mou```

# %% papermill={"duration": 0.06318, "end_time": "2021-08-13T07:16:41.298901", "exception": false, "start_time": "2021-08-13T07:16:41.235721", "status": "completed"}
ids = ['id','circle_id']
total_amounts = [i for i in list(data.columns) if re.search('total.+amt',i)]
total_outgoing_minutes = [i for i in list(data.columns) if re.search('total.+og.+mou',i)]
offnetwork_minutes = [i for i in list(data.columns) if re.search('offnet',i)]
average_revenue_3g = [i for i in list(data.columns) if re.search('arpu.+3g',i)]
average_revenue_2g = [i for i in list(data.columns) if re.search('arpu.+2g',i)]
volume_3g = [i for i in list(data.columns) if re.search('vol.+3g',i)]
volume_2g = [i for i in list(data.columns) if re.search('vol.+2g',i)]
age_on_network = [i for i in list(data.columns) if re.search('aon',i)]

#Storing them in a single flat list
variables = [*ids, 
             *total_amounts, 
             *total_outgoing_minutes, 
             *offnetwork_minutes, 
             *average_revenue_3g, 
             *average_revenue_2g,
             *volume_3g,
             *volume_2g,
             *age_on_network, 
             'churn_probability']

data = data[variables].set_index('id')

# %% papermill={"duration": 0.066079, "end_time": "2021-08-13T07:16:41.401769", "exception": false, "start_time": "2021-08-13T07:16:41.335690", "status": "completed"}
data.head()

# %% [markdown] papermill={"duration": 0.037694, "end_time": "2021-08-13T07:16:41.476864", "exception": false, "start_time": "2021-08-13T07:16:41.439170", "status": "completed"}
# Let's look at each variable's datatype:

# %% papermill={"duration": 0.064419, "end_time": "2021-08-13T07:16:41.578526", "exception": false, "start_time": "2021-08-13T07:16:41.514107", "status": "completed"}
data.info(verbose=1)

# %% [markdown] papermill={"duration": 0.037487, "end_time": "2021-08-13T07:16:41.653911", "exception": false, "start_time": "2021-08-13T07:16:41.616424", "status": "completed"}
# Let's also summarize the features using the df.describe method:

# %% papermill={"duration": 0.160583, "end_time": "2021-08-13T07:16:41.852218", "exception": false, "start_time": "2021-08-13T07:16:41.691635", "status": "completed"}
data.describe(include="all")

# %% [markdown] papermill={"duration": 0.03869, "end_time": "2021-08-13T07:16:41.929599", "exception": false, "start_time": "2021-08-13T07:16:41.890909", "status": "completed"}
# # 2. Create X, y and then Train test split
#
# Lets create X and y datasets and skip "circle_id" since it has only 1 unique value

# %% papermill={"duration": 0.04928, "end_time": "2021-08-13T07:16:42.017029", "exception": false, "start_time": "2021-08-13T07:16:41.967749", "status": "completed"}
data['circle_id'].unique()

# %% papermill={"duration": 0.054463, "end_time": "2021-08-13T07:16:42.111249", "exception": false, "start_time": "2021-08-13T07:16:42.056786", "status": "completed"}
X = data.drop(['circle_id'],1).iloc[:,:-1]
y = data.iloc[:,-1]

X.shape, y.shape

# %% [markdown] papermill={"duration": 0.039731, "end_time": "2021-08-13T07:16:42.189842", "exception": false, "start_time": "2021-08-13T07:16:42.150111", "status": "completed"}
# Splitting train and test data to avoid any contamination of the test data

# %% papermill={"duration": 0.065525, "end_time": "2021-08-13T07:16:42.294433", "exception": false, "start_time": "2021-08-13T07:16:42.228908", "status": "completed"}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

# %% papermill={"duration": 0.068615, "end_time": "2021-08-13T07:16:42.402101", "exception": false, "start_time": "2021-08-13T07:16:42.333486", "status": "completed"}
X_train.head()

# %% [markdown] papermill={"duration": 0.039575, "end_time": "2021-08-13T07:16:42.482838", "exception": false, "start_time": "2021-08-13T07:16:42.443263", "status": "completed"}
# # 3. Handling Missing data
#
# First lets analyse the missing data. We can use missingno library for quick visualizations.

# %% papermill={"duration": 1.699725, "end_time": "2021-08-13T07:16:44.222465", "exception": false, "start_time": "2021-08-13T07:16:42.522740", "status": "completed"}
msno.bar(X_train)

# %% papermill={"duration": 0.995529, "end_time": "2021-08-13T07:16:45.259912", "exception": false, "start_time": "2021-08-13T07:16:44.264383", "status": "completed"}
msno.matrix(X_train)

# %% [markdown] papermill={"duration": 0.043599, "end_time": "2021-08-13T07:16:45.347986", "exception": false, "start_time": "2021-08-13T07:16:45.304387", "status": "completed"}
# Lets also calculate the % missing data for each column:

# %% papermill={"duration": 0.06183, "end_time": "2021-08-13T07:16:45.454014", "exception": false, "start_time": "2021-08-13T07:16:45.392184", "status": "completed"}
missing_data_percent = 100*X_train.isnull().sum()/len(y_train)
missing_data_percent

# %% [markdown] papermill={"duration": 0.043963, "end_time": "2021-08-13T07:16:45.542693", "exception": false, "start_time": "2021-08-13T07:16:45.498730", "status": "completed"}
# Since too much missing information would make a column not really a great predictor for churn, we drop these columns and keep only the ones which have less than 40% missing data.

# %% papermill={"duration": 0.054309, "end_time": "2021-08-13T07:16:45.641625", "exception": false, "start_time": "2021-08-13T07:16:45.587316", "status": "completed"}
new_vars = missing_data_percent[missing_data_percent.le(40)].index
new_vars

# %% papermill={"duration": 0.05574, "end_time": "2021-08-13T07:16:45.742430", "exception": false, "start_time": "2021-08-13T07:16:45.686690", "status": "completed"}
X_train_filtered = X_train[new_vars]
X_train_filtered.shape

# %% [markdown] papermill={"duration": 0.045444, "end_time": "2021-08-13T07:16:45.832903", "exception": false, "start_time": "2021-08-13T07:16:45.787459", "status": "completed"}
# Next, we try imputation on variables with any amount of missing data still left. There are multiple ways of imputing data, and each will require a good business understanding of what the missing data is and how you may handle it.
#
# Some tips while working with missing data - 
#
# 1. Can simply replace missing values directly with a constant value such as 0
# 2. In certain cases you may want to replace it with the average value for each column respectively
# 3. For timeseries data, you may consider using linear or spline interplolation between a set of points, if you have data available for some of the months, and missing for the others.
# 4. You can consider more advance methods for imputation such as MICE.
#
# In our case, I will just demostrate a simple imputation with constant values as zeros.

# %% papermill={"duration": 0.056957, "end_time": "2021-08-13T07:16:45.935041", "exception": false, "start_time": "2021-08-13T07:16:45.878084", "status": "completed"}
missing_data_percent = X_train_filtered.isnull().any()
impute_cols = missing_data_percent[missing_data_percent.gt(0)].index
impute_cols

# %% papermill={"duration": 0.085462, "end_time": "2021-08-13T07:16:46.065826", "exception": false, "start_time": "2021-08-13T07:16:45.980364", "status": "completed"}
imp = SimpleImputer(strategy='constant', fill_value=0)
X_train_filtered[impute_cols] = imp.fit_transform(X_train_filtered[impute_cols])

# %% papermill={"duration": 0.914023, "end_time": "2021-08-13T07:16:47.025018", "exception": false, "start_time": "2021-08-13T07:16:46.110995", "status": "completed"}
msno.bar(X_train_filtered)

# %% papermill={"duration": 0.132076, "end_time": "2021-08-13T07:16:47.205627", "exception": false, "start_time": "2021-08-13T07:16:47.073551", "status": "completed"}
X_train_filtered.describe()

# %% [markdown] papermill={"duration": 0.047951, "end_time": "2021-08-13T07:16:47.301731", "exception": false, "start_time": "2021-08-13T07:16:47.253780", "status": "completed"}
# # 4. Exploratory Data Analysis & Preprocessing
#
# Lets start by analysing the univariate distributions of each feature.

# %% papermill={"duration": 0.885464, "end_time": "2021-08-13T07:16:48.234798", "exception": false, "start_time": "2021-08-13T07:16:47.349334", "status": "completed"}
plt.figure(figsize=(15,8))
plt.xticks(rotation=45)
sns.boxplot(data = X_train_filtered)


# %% [markdown] papermill={"duration": 0.050148, "end_time": "2021-08-13T07:16:48.336611", "exception": false, "start_time": "2021-08-13T07:16:48.286463", "status": "completed"}
# ### 4.1 Handling outliers
#
# The box plots of these features show there a lot of outliers. These can be capped with k-sigma method.

# %% papermill={"duration": 0.058225, "end_time": "2021-08-13T07:16:48.444315", "exception": false, "start_time": "2021-08-13T07:16:48.386090", "status": "completed"}
def cap_outliers(array, k=3):
    upper_limit = array.mean() + k*array.std()
    lower_limit = array.mean() - k*array.std()
    array[array<lower_limit] = lower_limit
    array[array>upper_limit] = upper_limit
    return array


# %% papermill={"duration": 0.995019, "end_time": "2021-08-13T07:16:49.488905", "exception": false, "start_time": "2021-08-13T07:16:48.493886", "status": "completed"}
X_train_filtered1 = X_train_filtered.apply(cap_outliers, axis=0)

plt.figure(figsize=(15,8))
plt.xticks(rotation=45)
sns.boxplot(data = X_train_filtered1)

# %% [markdown] papermill={"duration": 0.050821, "end_time": "2021-08-13T07:16:49.590800", "exception": false, "start_time": "2021-08-13T07:16:49.539979", "status": "completed"}
# ### 4.2 Feature scaling
#
# Lets also scale the features by scaling them with Standard scaler (few other alternates are min-max scaling and Z-scaling).

# %% papermill={"duration": 0.077522, "end_time": "2021-08-13T07:16:49.720148", "exception": false, "start_time": "2021-08-13T07:16:49.642626", "status": "completed"}
scale = StandardScaler()
X_train_filtered2 = scale.fit_transform(X_train_filtered1)

# %% papermill={"duration": 0.893696, "end_time": "2021-08-13T07:16:50.665247", "exception": false, "start_time": "2021-08-13T07:16:49.771551", "status": "completed"}
plt.figure(figsize=(15,8))
plt.xticks(rotation=45)
sns.boxplot(data = pd.DataFrame(X_train_filtered2, columns=new_vars))

# %% [markdown] papermill={"duration": 0.053477, "end_time": "2021-08-13T07:16:50.772231", "exception": false, "start_time": "2021-08-13T07:16:50.718754", "status": "completed"}
# You can perform feature transformations at this stage. 
#
# 1. **Positively skewed:** Common transformations of this data include square root, cube root, and log.
# 2. **Negatively skewed:** Common transformations include square, cube root and logarithmic.
#
# Please read the following link to understand how to perform feature scaling and preprocessing : https://scikit-learn.org/stable/modules/preprocessing.html
#  
# Lets also plot the correlations for each feature for bivariate analysis.

# %% papermill={"duration": 0.568818, "end_time": "2021-08-13T07:16:51.394261", "exception": false, "start_time": "2021-08-13T07:16:50.825443", "status": "completed"}
plt.figure(figsize=(10,8))
sns.heatmap(pd.DataFrame(X_train_filtered2, columns=new_vars).corr())

# %% papermill={"duration": 0.393552, "end_time": "2021-08-13T07:16:51.843093", "exception": false, "start_time": "2021-08-13T07:16:51.449541", "status": "completed"}
#Distribution for the churn probability
sns.histplot(y_train)

# %% [markdown] papermill={"duration": 0.054582, "end_time": "2021-08-13T07:16:51.952415", "exception": false, "start_time": "2021-08-13T07:16:51.897833", "status": "completed"}
# # 5. Feature engineering and selection
#
# Let's understand feature importances for raw features as well as components to decide top features for modelling.

# %% papermill={"duration": 8.035233, "end_time": "2021-08-13T07:17:00.041999", "exception": false, "start_time": "2021-08-13T07:16:52.006766", "status": "completed"}
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
rf.fit(X_train_filtered2, y_train)

# %% papermill={"duration": 0.165618, "end_time": "2021-08-13T07:17:00.262949", "exception": false, "start_time": "2021-08-13T07:17:00.097331", "status": "completed"}
feature_importances = pd.DataFrame({'col':new_vars, 'importance':rf.feature_importances_})

# %% papermill={"duration": 0.295789, "end_time": "2021-08-13T07:17:00.614769", "exception": false, "start_time": "2021-08-13T07:17:00.318980", "status": "completed"}
plt.figure(figsize=(15,8))
plt.xticks(rotation=45)
plt.bar(feature_importances['col'], feature_importances['importance'])

# %% [markdown] papermill={"duration": 0.055776, "end_time": "2021-08-13T07:17:00.727093", "exception": false, "start_time": "2021-08-13T07:17:00.671317", "status": "completed"}
# At this step, you can create a bunch of features based on business understanding, such as 
# 1. "average % gain of 3g volume from month 6 to 8" - (growth or decline of 3g usage month over month?)
# 2. "ratio of total outgoing amount and age of user on network" - (average daily usage of a user?)
# 3. "standard deviation of the total amount paid by user for all services" - (too much variability in charges?)
# 4. etc..
#
# Another way of finding good features would be to project them into a lower dimensional space using PCA. PCA creates components which are a linear combination of the features. This then allows you to select components which explain the highest amount of variance.
#
# Lets try to project the data onto 2D space and plot. **Note:** you can try TSNE, which is another dimensionality reduction approach as well. Check https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html for moree details.

# %% papermill={"duration": 4.617268, "end_time": "2021-08-13T07:17:05.400295", "exception": false, "start_time": "2021-08-13T07:17:00.783027", "status": "completed"}
pca = PCA()
pca_components = pca.fit_transform(X_train_filtered2)
sns.scatterplot(x=pca_components[:,0], y=pca_components[:,1], hue=y_train)

# %% papermill={"duration": 2.631052, "end_time": "2021-08-13T07:17:08.093002", "exception": false, "start_time": "2021-08-13T07:17:05.461950", "status": "completed"}
sns.scatterplot(x=pca_components[:,1], y=pca_components[:,2], hue=y_train)

# %% [markdown] papermill={"duration": 0.063551, "end_time": "2021-08-13T07:17:08.224795", "exception": false, "start_time": "2021-08-13T07:17:08.161244", "status": "completed"}
# Let's also check which of the components have high feature importances towards the end goal of churn prediction.

# %% papermill={"duration": 12.694325, "end_time": "2021-08-13T07:17:20.982792", "exception": false, "start_time": "2021-08-13T07:17:08.288467", "status": "completed"}
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
rf.fit(pca_components, y_train)

feature_importances = pd.DataFrame({'col':['component_'+str(i) for i in range(16)], 
                                    'importance':rf.feature_importances_})

plt.figure(figsize=(15,8))
plt.xticks(rotation=45)
plt.bar(feature_importances['col'], feature_importances['importance'])

# %% [markdown] papermill={"duration": 0.065189, "end_time": "2021-08-13T07:17:21.113066", "exception": false, "start_time": "2021-08-13T07:17:21.047877", "status": "completed"}
# # 6. Model building
#
# Let's build a quick model with logistic regression and the first 2 PCA components.

# %% papermill={"duration": 0.185674, "end_time": "2021-08-13T07:17:21.364082", "exception": false, "start_time": "2021-08-13T07:17:21.178408", "status": "completed"}
lr = LogisticRegression(max_iter=1000, tol=0.001, solver='sag')
lr.fit(pca_components[:,:2], y_train)

# %% papermill={"duration": 0.089994, "end_time": "2021-08-13T07:17:21.519838", "exception": false, "start_time": "2021-08-13T07:17:21.429844", "status": "completed"}
lr.score(pca_components[:,:2], y_train)

# %% [markdown] papermill={"duration": 0.064818, "end_time": "2021-08-13T07:17:21.699674", "exception": false, "start_time": "2021-08-13T07:17:21.634856", "status": "completed"}
# The model has 89.8% accuracy, but let's build a pipeline to fit and score the model faster.
#
# The steps of this pipeline would be the following, but this is only one type of pipeline -
# 1. Imputation
# 2. Scaling
# 3. PCA
# 4. Classification model
#
# You can change this pipeline, add addition transformations, change models, use cross validation or even use this pipeline to work with a Gridsearch.

# %% papermill={"duration": 0.071699, "end_time": "2021-08-13T07:17:21.836592", "exception": false, "start_time": "2021-08-13T07:17:21.764893", "status": "completed"}
imp = SimpleImputer(strategy='constant', fill_value=0)
scale = StandardScaler()
pca = PCA(n_components=10)
lr = LogisticRegression(max_iter=1000, tol=0.001)

# %% papermill={"duration": 0.073281, "end_time": "2021-08-13T07:17:21.975259", "exception": false, "start_time": "2021-08-13T07:17:21.901978", "status": "completed"}
pipe = Pipeline(steps = [('imputation',imp),
                         ('scaling',scale),
                         ('pca',pca),
                         ('model',lr)])

# %% papermill={"duration": 0.485922, "end_time": "2021-08-13T07:17:22.526693", "exception": false, "start_time": "2021-08-13T07:17:22.040771", "status": "completed"}
pipe.fit(X_train[new_vars], y_train)

# %% papermill={"duration": 0.111563, "end_time": "2021-08-13T07:17:22.755300", "exception": false, "start_time": "2021-08-13T07:17:22.643737", "status": "completed"}
train_score = pipe.score(X_train[new_vars], y_train)
print("Training accuracy:", train_score)

# %% papermill={"duration": 0.087891, "end_time": "2021-08-13T07:17:22.960566", "exception": false, "start_time": "2021-08-13T07:17:22.872675", "status": "completed"}
test_score = pipe.score(X_test[new_vars], y_test)
print("Test accuracy:", test_score)

# %% [markdown] papermill={"duration": 0.070399, "end_time": "2021-08-13T07:17:23.152879", "exception": false, "start_time": "2021-08-13T07:17:23.082480", "status": "completed"}
# Let's make a confusion matrix to analyze how each class is being predicted by the model.

# %% papermill={"duration": 0.200569, "end_time": "2021-08-13T07:17:23.421831", "exception": false, "start_time": "2021-08-13T07:17:23.221262", "status": "completed"}
confusion_matrix(y_train, pipe.predict(X_train[new_vars]))

# %% papermill={"duration": 0.109186, "end_time": "2021-08-13T07:17:23.608435", "exception": false, "start_time": "2021-08-13T07:17:23.499249", "status": "completed"}
confusion_matrix(y_test, pipe.predict(X_test[new_vars]))

# %% [markdown] papermill={"duration": 0.068268, "end_time": "2021-08-13T07:17:23.788669", "exception": false, "start_time": "2021-08-13T07:17:23.720401", "status": "completed"}
# We can see a high amount of type 2 error. Due to class imbalance, the model is clearly trying to predict majority of the cases as class 0. Understanding how to handle class imbalance in classification models might be the key to winning this competition :) (hint!)

# %% papermill={"duration": 0.09953, "end_time": "2021-08-13T07:17:23.955991", "exception": false, "start_time": "2021-08-13T07:17:23.856461", "status": "completed"}
precision_score(y_test, pipe.predict(X_test[new_vars]))

# %% papermill={"duration": 0.123943, "end_time": "2021-08-13T07:17:24.198308", "exception": false, "start_time": "2021-08-13T07:17:24.074365", "status": "completed"}
recall_score(y_test, pipe.predict(X_test[new_vars]))

# %% [markdown] papermill={"duration": 0.067672, "end_time": "2021-08-13T07:17:24.385001", "exception": false, "start_time": "2021-08-13T07:17:24.317329", "status": "completed"}
# # 7. Creating submission file
#
# For submission, we need to make sure that the format is exactly the same as the sample.csv file. It contains 2 columns, id and churn_probability

# %% papermill={"duration": 0.080814, "end_time": "2021-08-13T07:17:24.533810", "exception": false, "start_time": "2021-08-13T07:17:24.452996", "status": "completed"}
sample.head()

# %% [markdown] papermill={"duration": 0.068381, "end_time": "2021-08-13T07:17:24.670628", "exception": false, "start_time": "2021-08-13T07:17:24.602247", "status": "completed"}
# The submission file should contain churn_probability values that have to be predicted for the unseen data provided (test.csv)

# %% papermill={"duration": 0.098877, "end_time": "2021-08-13T07:17:24.838199", "exception": false, "start_time": "2021-08-13T07:17:24.739322", "status": "completed"}
unseen.head()

# %% [markdown] papermill={"duration": 0.06952, "end_time": "2021-08-13T07:17:24.977086", "exception": false, "start_time": "2021-08-13T07:17:24.907566", "status": "completed"}
# Lets first select the columns that we want to work with (or create them, if you have done any feature engineering)

# %% papermill={"duration": 0.093498, "end_time": "2021-08-13T07:17:25.140420", "exception": false, "start_time": "2021-08-13T07:17:25.046922", "status": "completed"}
submission_data = unseen.set_index('id')[new_vars]
submission_data.shape

# %% [markdown] papermill={"duration": 0.069588, "end_time": "2021-08-13T07:17:25.279867", "exception": false, "start_time": "2021-08-13T07:17:25.210279", "status": "completed"}
# Next, lets create a new column in the unseen dataset called churn_probability and use the model pipeline to predict the probabilities for this data

# %% papermill={"duration": 0.108369, "end_time": "2021-08-13T07:17:25.457668", "exception": false, "start_time": "2021-08-13T07:17:25.349299", "status": "completed"}
unseen['churn_probability'] = pipe.predict(submission_data)
output = unseen[['id','churn_probability']]
output.head()

# %% [markdown] papermill={"duration": 0.124336, "end_time": "2021-08-13T07:17:25.701056", "exception": false, "start_time": "2021-08-13T07:17:25.576720", "status": "completed"}
# Finally, lets create a csv file out of this dataset, ensuring to set index=False to avoid an addition column in the csv.

# %% papermill={"duration": 0.12837, "end_time": "2021-08-13T07:17:25.914318", "exception": false, "start_time": "2021-08-13T07:17:25.785948", "status": "completed"}
output.to_csv('submission_pca_lr_13jul.csv',index=False)

# %% [markdown] papermill={"duration": 0.08214, "end_time": "2021-08-13T07:17:26.080507", "exception": false, "start_time": "2021-08-13T07:17:25.998367", "status": "completed"}
# You can now take this file and upload it as a submission on Kaggle.

# %% papermill={"duration": 0.070782, "end_time": "2021-08-13T07:17:26.223247", "exception": false, "start_time": "2021-08-13T07:17:26.152465", "status": "completed"}
