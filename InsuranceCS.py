#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scikitplot as skplt
import sklearn

from sklearn.preprocessing    import MinMaxScaler, StandardScaler
from sklearn import model_selection, ensemble, neighbors, linear_model
from sklearn.metrics          import roc_auc_score
from sklearn.ensemble         import RandomForestClassifier
from sklearn.naive_bayes      import GaussianNB
from xgboost                  import XGBClassifier
from lightgbm                 import LGBMClassifier
from catboost                 import CatBoostClassifier

from sklearn.model_selection  import StratifiedKFold

import inflection

from IPython.display            import Image
from IPython.core.display       import HTML

import warnings
warnings.filterwarnings("ignore")


# ## Helper Functions

# In[2]:


## performance metrics

# definition of precision_at_k for the top 20.000 clients as default
def precision_at_k (data, k=2000):
    
    # reset index
    data = data.reset_index(drop=True)
    
    #create ranking order
    data['ranking'] = data.index + 1
    
    #calculate precision based on column named response
    data['precision_at_k'] = data['response'].cumsum() / data['ranking']
    
    return data.loc[k, 'precision_at_k']

# definition of recall_at_k for the top 20.000 clients as default
def recall_at_k (data, k=20000):
    
    # reset index
    data = data.reset_index(drop=True)
    
    #create ranking order
    data['ranking'] = data.index + 1
    
    #calculate recall based on the sum of responses
    data['recall_at_k'] = data['response'].cumsum() / data['response'].sum()
    
    return data.loc[k, 'recall_at_k']

##Define models accuracy function
def accuracy (model_name, x_val, y_val, yhat):
    
    data = x_val.copy()
    data['response'] = y_val.copy()
    data['score'] = yhat[:, 1].tolist()
    data = data.sort_values('score', ascending=False)
   
    precision = precision_at_k(data)      
    recall = recall_at_k(data)
    f1_score = round(2*(precision * recall) / (precision + recall), 3)
    roc = roc_auc_score(y_val, yhat[:,1])
    
    return pd.DataFrame({'Model Name': model_name,
                         'ROC AUC': roc.round(4),
                         'Precision@K Mean': np.mean(precision).round(4),
                         'Recall@K Mean': np.mean(recall).round(4),
                         'F1_Score' : np.mean(f1_score).round(4)}, index=[0])

def jupyter_settings():
    get_ipython().run_line_magic('matplotlib', 'inline')
    get_ipython().run_line_magic('pylab', 'inline')
    
    plt.style.use( 'bmh' )
    plt.rcParams['figure.figsize'] = [25, 12]
    plt.rcParams['font.size'] = 24
    
    display( HTML( '<style>.container { width:100% !important; }</style>') )
    pd.options.display.max_columns = None
    pd.options.display.max_rows = None
    pd.set_option( 'display.expand_frame_repr', False )
    
    sns.set()

jupyter_settings();


# ## Loading Data 

# In[3]:


df_raw = pd.read_csv('data/train.csv')


# # Data Details

# In[4]:


df2 = df_raw.copy()


# In[5]:


df2.head()


# ## Data Dictionary

# |The data set that we're using is from Kaggle (https://www.kaggle.com/anmolkumar/health-insurance-cross-sell-prediction).
# 
# 
# 
# | Feature                                       |Description   
# |:---------------------------|:---------------
# | **Id**                         | Unique ID for the customer   | 
# | **Gender**                           | Gender of the customer   | 
# | **Driving License**                                   | 0, customer does not have DL; 1, customer already has DL  | 
# | **Region Code**                               | Unique code for the region of the customer   | 
# | **Previously Insured**                     | 1, customer already has vehicle insurance; 0, customer doesn't have vehicle insurance | 
# | **Vehicle Age**                     | Age of the vehicle | 
# | **Vehicle Damage**                                  | 1, customer got his/her vehicle damaged in the past; 0, customer didn't get his/her vehicle damaged in the past | 
# | **Anual Premium**                             | The amount customer needs to pay as premium in the year | 
# | **Policy sales channel**                                    | Anonymized Code for the channel of outreaching to the customer ie  | 
# | **Vintage**                | Number of Days, customer has been associated with the company  | 
# | **Response**              | 1, customer is interested; 0, customer is not interested. |    

# ## Rename Columns

# In[6]:


cols_old = ['id', 'Gender', 'Age', 'Driving_License', 'Region_Code', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage', 'Annual_Premium', 
            'Policy_Sales_Channel', 'Vintage', 'Response']

snakecase = lambda x: inflection.underscore( x )

cols_new = list( map( snakecase, cols_old ) )

# rename
df2.columns = cols_new


# ## Data Dimensions

# In[7]:


print ('Number of Rows: {}'.format( df2.shape[0]))
print ('Number of Columns: {}'.format( df2.shape[1]))


# ## Data Types

# In[8]:


df2.dtypes


# ## Missing Values

# In[9]:


df2.isna().sum()


# ## Change Types

# In[10]:


# changing data types from float to int64

df2['region_code'] = df2['region_code'].astype('int64')     

df2['policy_sales_channel'] = df2['policy_sales_channel'].astype('int64')    

df2['annual_premium'] = df2['annual_premium'].astype('int64')    


# ## Descriptive Statistics
# 

# In[11]:


# Split numerical and categorical features
num_attributes = df2.select_dtypes( include=['int64', 'float64'])
cat_attributes = df2.select_dtypes( exclude=['int64', 'float64'])


# ### Numerical Attributes

# In[12]:


#Central Tendency - mean, meadian
ct1 = pd.DataFrame( num_attributes.apply( np.mean ) ).T
ct2 = pd.DataFrame( num_attributes.apply( np.median ) ).T

# dispersion - std, min, max, range, skew, kurtosis
d1 = pd.DataFrame( num_attributes.apply( np.std ) ).T
d2 = pd.DataFrame( num_attributes.apply( min ) ).T
d3 = pd.DataFrame( num_attributes.apply( max ) ).T
d4 = pd.DataFrame( num_attributes.apply( lambda x: x.max() - x.min() ) ).T
d5 = pd.DataFrame( num_attributes.apply( lambda x: x.skew() ) ).T
d6 = pd.DataFrame( num_attributes.apply( lambda x: x.kurtosis() ) ).T

# concat
m= pd.concat( [d2, d3, d4, ct1, ct2, d1, d5, d6] ).T.reset_index()
m.columns = ['attributes','min', 'max', 'range', 'mean', 'median', 'std', 'skew', 'kurtosis']
m


# **Age** of customers ranges from 20 to 85 years old, average being close to 38.
# 
# **Driving Licence** ≈ 100% of the clients in analysis retain a one
# 
# **Vehicle insurance** ≈ 55% of the clients do not hold one
# 
# **Annual Premium** Clients pay ≈ 30.5k on their current health insurance policy
# 
# **Response** 12.23% of the clients showed to be interest in purchasing a vehicle insurance.

# ### Categorical Attributes

# In[13]:


# add percentage of most common attribute
cat_attributes_p = cat_attributes.describe().T
cat_attributes_p['freq_p'] = cat_attributes_p['freq'] / cat_attributes_p['count']
cat_attributes_p


# **Gender** ≈ 54% of the customers are Male
# 
# **Vehicle Age** Customers age vehicle is most commonly between 1 and 2 years old
# 
# **Vehicle Damage** ≈ 50% of the customers got his/her vehicle damaged in the past

# **TOP 3 Combos Categorical Attributes**

# In[14]:


categorical_combo = pd.DataFrame(round(cat_attributes.value_counts(normalize=True) * 100)).reset_index().rename(columns={0: '%'})
categorical_combo['count'] = cat_attributes.value_counts().values
display(categorical_combo)


# **1.** Males with car age between 1-2 years old that got vehicle damaged in the past 
# 
# **2.** Female with car newer than 1 year old that never got vehicle damage in the past
# 
# **3.** Males with car newer than 1 year old that never got vehicle damage in the past

# # Feature Engineering

# In[15]:


df3 = df2.copy()


# ## Features Creation 

# In[16]:


# vehicle age
df3['vehicle_age']= df3['vehicle_age'].apply( lambda x: 'over_2_years' if x == '> 2 Years' else 'between_1_2year' if x== '1-2 Year' else 'between_1_2_year')
# vehicle damage
df3['vehicle_damage'] = df3['vehicle_damage'].apply( lambda x: 1 if x == 'Yes' else 0 )


# ## Mind Map

# # Exploratory Data Analysis (EDA)

# In[17]:


df4 = df3.copy()


# ## Univariate Analysis

# ### Response Variable

# In[18]:


sns.countplot(x = 'response', data=df4);


# ### Numerical Variables

# In[19]:


num_attributes.hist(bins=25);


# #### Age
# 
# **Findings** The average age of interested clients is higher than non-interested clients. Both plots disclose well how younger clients are not as interested as older clients.

# In[20]:


plt.subplot(1, 2, 1)
sns.boxplot( x='response', y='age', data=df4 )

plt.subplot(1, 2, 2)
sns.histplot(df4, x='age', hue='response');


# #### Driving Licence
# **Findings** Only clients holding a driving license are part of the dataset. 12% are potential vehicle insurance customers

# In[21]:


aux2 = pd.DataFrame(round(df4[['driving_license', 'response']].value_counts(normalize=True) * 100)).reset_index().rename(columns={0: '%'})
aux2['count'] = (aux2['%'] * df4.shape[0]).astype(int)
aux2


# In[22]:


aux2 = df4[['driving_license', 'response']].groupby( 'response' ).sum().reset_index()
sns.barplot( x='response', y='driving_license', data=aux2 );


# #### Region Code

# In[23]:


aux3 = df4[['id', 'region_code', 'response']].groupby( ['region_code', 'response'] ).count().reset_index()
aux3 = aux3[(aux3['id'] > 1000) & (aux3['id'] < 20000)]
sns.scatterplot( x='region_code', y='id', hue='response', data=aux3 );


# #### Previously Insured
# **Findings** All potential vehicle insurance customers have never held an insurance. 46% of our clients already have vehicle insurance and are not interested.

# In[24]:


aux4 = pd.DataFrame(round(df4[['previously_insured', 'response']].value_counts(normalize=True) * 100)).reset_index().rename(columns={0: '%'})
aux4['count'] = (aux4['%'] * df4.shape[0]).astype(int)
aux4


# In[25]:


sns.barplot(data=aux4, x='previously_insured', y='count', hue='response');


# #### Annual Premium
# **Findings** Annual premiums for both interested and non-interested clients are very similar.

# In[26]:


aux5 = df4[(df4['annual_premium'] <100000)]
sns.boxplot( x='response', y='annual_premium', data=aux5 );


# #### Policy Sales Channel
# **Findings**

# In[27]:


aux6 = df4[['policy_sales_channel', 'response']].groupby( 'policy_sales_channel' ).sum().reset_index()

plt.xticks(rotation=90)
ax6 = sns.barplot( x='response', y='policy_sales_channel', data=aux6, order = aux6['response']);


# In[28]:


aux6 = df4[['policy_sales_channel', 'response']].groupby( 'policy_sales_channel' ).sum().reset_index()

plt.xticks(rotation=90)
ax6 = sns.barplot( x='response', y='policy_sales_channel', data=aux6, order = aux6['response']);


# #### Vintage
# **Findings**

# In[29]:


plt.subplot(1, 2, 1)
sns.boxplot( x='response', y='vintage', data=df4 )

plt.subplot(1, 2, 2)
sns.histplot(df4, x='vintage', hue='response');


# ### Categorical Variables

# #### Gender

# In[30]:


aux7 = pd.DataFrame(round(df4[['gender', 'response']].value_counts(normalize=True) * 100)).reset_index().rename(columns={0: '%'})
aux7['count'] = (aux7['%'] * df4.shape[0]).astype(int)
aux7


# In[31]:


sns.barplot(data=aux7, x='gender', y='count', hue='response');


# #### Vehicle Age

# In[32]:


aux8 = pd.DataFrame(round(df4[['vehicle_age', 'response']].value_counts(normalize=True) * 100)).reset_index().rename(columns={0: '%'})
aux8['count'] = (aux8['%'] * df4.shape[0]).astype(int)
aux8


# In[33]:


sns.barplot(data=aux8, x='vehicle_age', y='count', hue='response');


# #### Vehicle Damage

# In[34]:


aux9 = pd.DataFrame(round(df4[['vehicle_damage', 'response']].value_counts(normalize=True) * 100)).reset_index().rename(columns={0: '%'})
aux9['count'] = (aux9['%'] * df4.shape[0]).astype(int)
aux9


# In[35]:


sns.barplot(data=aux9, x='vehicle_damage', y='count', hue='response');


# ## Bivariate Analysis

# ## Multivariate Analysis

# ### Numerical Attributed
# **Finding** Having the target variable in scope, the stronger correlations with feature 'Previously Insured' (-0.34), 'Policy Sales Channel' (-0.14) and 'Age' (0.11). Outside the target variable scope, between Age and Policy Sales Chanel there is strong negative correlation of -0.58), 'Previously Insured' and 'Age' of -0.25 and last between 'Previously Insured' and 'Policy Sales Channel' 0.22. 

# In[36]:


corr_matrix= num_attributes.corr()
# Half matrix
mask = np.zeros_like(corr_matrix)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr_matrix, mask = mask, annot = True, square = True, cmap='YlGnBu');


# # Data Preparation

# In[37]:


df5 = df4.copy()


# ## Standardization of DataSets 

# In[38]:


df5['annual_premium'] = StandardScaler().fit_transform( df5[['annual_premium']].values)


# ## Rescaling

# In[39]:


mms = MinMaxScaler()

#age
df5['age'] = mms.fit_transform( df5[['age']].values)

#vintage
df5['vintage'] = mms.fit_transform( df5[['vintage']].values)


# ## Transformation

# ### Encoding

# In[40]:


#gender - target encoder
target_encode_gender = df5.groupby('gender')['response'].mean()
df5.loc[:, 'gender'] = df5['gender'].map(target_encode_gender)

# region_code - Target Encoding - as there are plenty of categories (as seen in EDA) it is better not to use one hot encoding and to use 
target_encode_region_code = df5.groupby('region_code')['response'].mean()
df5.loc[:, 'region_code'] = df5['region_code'].map(target_encode_region_code)

#vehicle_age
df5 = pd.get_dummies( df5, prefix='vehicle_age', columns=['vehicle_age'] )

#policy_sales_channel - Frequency encode
fe_policy_sales_channel = df5.groupby('policy_sales_channel').size()/len( df5)
df5['policy_sales_channel'] = df5['policy_sales_channel'].map(fe_policy_sales_channel)


# # Feature Selection

# In[41]:


df6 = df5.copy()


# ## Split dataframe into training and test

# In[42]:


X = df6.drop('response', axis=1)
y = df6['response'].copy()

x_train, x_val, y_train, y_val = model_selection.train_test_split(X, y, test_size = 0.2)

df6 = pd.concat ( [x_train, y_train], axis = 1)


# ## Feature Importance

# In[43]:


forest = ensemble.ExtraTreesClassifier( n_estimators = 250, random_state = 42, n_jobs = -1)

x_train_n = df6.drop(['id', 'response'], axis=1 )
y_train_n = y_train.values
forest.fit( x_train_n, y_train_n)


# In[44]:


importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis = 0)
indices = np.argsort(importances)[::-1]
#print the feature ranking
print( "Feature Rankings")
df = pd.DataFrame()
for i, j in zip(x_train_n, forest.feature_importances_):
    aux = pd.DataFrame( {'feature': i, 'importance':j}, index=[0])
    df = pd.concat ([df, aux], axis = 0)
    
print( df.sort_values( 'importance', ascending=False))

# PLt the impurity-based feature importance of the features
plt.figure()
plt.title('Feature Importance')
plt.bar(range(x_train_n.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
plt.xticks(range(x_train_n.shape[1]), indices)
plt.xlim([-1, x_train_n.shape[1]])
plt.show()


# # Machine Learning Modelling

# In[45]:


cols_selected = ['vintage', 'annual_premium', 'age', 'region_code', 
                 'vehicle_damage', 'policy_sales_channel']

cols_not_selected = ['previously_insured', 'vehicle_age_between_1_2_year', 'vehicle_age_between_1_2year', 'gender', 'vehicle_age_over_2_years', 'driving_license']

#create df to be used for business understading
x_validation = x_val.drop(cols_not_selected, axis=1)

#create dfs for modeling
x_train = df6[cols_selected]
x_val = x_val[cols_selected]


# ## Logistic Regression

# ### Model Building

# In[46]:


#define model
lr = linear_model.LogisticRegression (random_state = 42)

#train model
lr.fit( x_train, y_train)

#model prediction
yhat_lr = lr.predict_proba( x_val)


# ### Model Single Performance

# In[47]:


accuracy_lr = accuracy('Linear Regression', x_val, y_val, yhat_lr)
accuracy_lr


# In[48]:


# Accumulative Gain
skplt.metrics.plot_cumulative_gain(y_val, yhat_lr);


# In[49]:


# Lift Curve
skplt.metrics.plot_lift_curve( y_val, yhat_lr );


# ## Naive Bayes

# ### Model Building

# In[50]:


#define model
naive = GaussianNB()

#train model
naive.fit( x_train, y_train)

#model prediction
yhat_naive = naive.predict_proba( x_val)


# ### Model Single Performance

# In[51]:


accuracy_naive = accuracy('Naive Bayes', x_val, y_val, yhat_naive)
accuracy_naive


# In[52]:


# Accumulative Gain
skplt.metrics.plot_cumulative_gain(y_val, yhat_naive);


# In[53]:


# Lift Curve
skplt.metrics.plot_lift_curve( y_val, yhat_naive );


# ## Extra Trees

# ### Model Building

# In[54]:


#define model
et = ensemble.ExtraTreesClassifier (random_state = 42, n_jobs=-1)

#train model
et.fit( x_train, y_train)

#model prediction
yhat_et = et.predict_proba( x_val)


# ### Model Single Performance

# In[55]:


accuracy_et = accuracy('Extra Trees Classifier', x_val, y_val, yhat_et)
accuracy_et


# In[56]:


# Accumulative Gain
skplt.metrics.plot_cumulative_gain(y_val, yhat_et);


# In[57]:


# Lift Curve
skplt.metrics.plot_lift_curve( y_val, yhat_et );


# ### Model Accuracy

# ## Random Forest Regressor

# ### Model Building

# In[58]:


#define model
rf=RandomForestClassifier(n_estimators=100, min_samples_leaf=25)

#train model
rf.fit( x_train, y_train)

#model prediction
yhat_rf = rf.predict_proba( x_val)


# ### Model Single Performance

# In[59]:


accuracy_rf = accuracy('Random Forest Regressor', x_val, y_val, yhat_rf)
accuracy_rf


# In[60]:


# Accumulative Gain
skplt.metrics.plot_cumulative_gain(y_val, yhat_rf);


# In[61]:


# Lift Curve
skplt.metrics.plot_lift_curve( y_val, yhat_rf );


# ## KNN Classifier

# ### Model Building

# In[62]:


#define model
knn = neighbors.KNeighborsClassifier (n_neighbors = 8)

#train model
knn.fit( x_train, y_train)

#model prediction
yhat_knn = knn.predict_proba( x_val)


# ### Model Single Performance

# In[63]:


accuracy_knn = accuracy('K-Nearest Neighbours', x_val, y_val, yhat_knn)
accuracy_knn


# In[64]:


# Accumulative Gain
skplt.metrics.plot_cumulative_gain(y_val, yhat_knn);


# In[65]:


# Lift Curve
skplt.metrics.plot_lift_curve( y_val, yhat_knn );


# ### Cross Validation

# ## XGBoost Classifier

# ### Model Building

# In[66]:


#define model
xgboost = XGBClassifier(objective='binary:logistic',
                        eval_metric='error',
                        n_estimators = 100,
                        random_state = 22)

#train model
xgboost.fit( x_train, y_train)

#model prediction
yhat_xgboost = xgboost.predict_proba( x_val)


# ### Model Single Performance

# In[67]:


accuracy_xgboost = accuracy('XGBoost Classifier', x_val, y_val, yhat_xgboost)
accuracy_xgboost


# In[68]:


# Accumulative Gain
skplt.metrics.plot_cumulative_gain(y_val, yhat_xgboost);


# In[69]:


# Lift Curve
skplt.metrics.plot_lift_curve( y_val, yhat_xgboost );


# ## LightGBM Classifier

# ### Model Building

# In[70]:


#define model
lgbm = LGBMClassifier(random_state = 22)

#train model
lgbm.fit( x_train, y_train)

#model prediction
yhat_lgbm = lgbm.predict_proba( x_val)


# ### Model Single Performance

# In[71]:


accuracy_lgbm = accuracy('LightGBM Classifier', x_val, y_val, yhat_lgbm)
accuracy_lgbm


# In[72]:


# Accumulative Gain
skplt.metrics.plot_cumulative_gain(y_val, yhat_lgbm);


# In[73]:


# Lift Curve
skplt.metrics.plot_lift_curve( y_val, yhat_lgbm );


# ## CatBoost Classifier

# ### Model Building

# In[74]:


#define model
catboost = CatBoostClassifier(verbose = False, random_state = 22)

#train model
catboost.fit( x_train, y_train)

#model prediction
yhat_catboost = catboost.predict_proba( x_val)


# ### Model Single Performance

# In[75]:


accuracy_catboost = accuracy('CatBoost Classifier', x_val, y_val, yhat_catboost)
accuracy_catboost


# In[76]:


# Accumulative Gain
skplt.metrics.plot_cumulative_gain(y_val, yhat_catboost);


# In[77]:


# Lift Curve
skplt.metrics.plot_lift_curve( y_val, yhat_catboost );


# # Hyperparameter Fine Tuning

# # Performance Evaluation and Interpretation

# In[ ]:




