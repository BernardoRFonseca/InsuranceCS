#!/usr/bin/env python
# coding: utf-8

# # 0 Imports

# In[180]:


import pandas                 as pd
import numpy                  as np
import seaborn                as sns
import matplotlib.pyplot      as plt
import scikitplot             as skplt
import sklearn
import inflection
import pickle

from sklearn.preprocessing    import MinMaxScaler, StandardScaler
from sklearn                  import model_selection
from sklearn.model_selection  import StratifiedKFold
from sklearn                  import ensemble
from sklearn                  import neighbors
from sklearn                  import linear_model
from sklearn.metrics          import roc_auc_score
from sklearn.ensemble         import RandomForestClassifier
from sklearn.naive_bayes      import GaussianNB
from xgboost                  import XGBClassifier
from lightgbm                 import LGBMClassifier
from catboost                 import CatBoostClassifier
from kds.metrics              import plot_cumulative_gain, plot_lift

from IPython.display          import Image
from IPython.core.display     import HTML

import warnings
warnings.filterwarnings("ignore")


# ## 0.1 Helper Functions

# ### 0.1.1 Models Performance

# In[2]:


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
def accuracy (model, x_val, y_val, yhat):
    
    data = x_val.copy()
    data['response'] = y_val.copy()
    data['score'] = yhat[:, 1].tolist()
    data = data.sort_values('score', ascending=False)
   
    precision = precision_at_k(data)      
    recall = recall_at_k(data)
    f1_score = round(2*(precision * recall) / (precision + recall), 3)
    roc = roc_auc_score(y_val, yhat[:,1])
    
    return pd.DataFrame({'Model Name': type(model).__name__,
                         'ROC AUC': roc.round(4),
                         'Precision@K Mean': np.mean(precision).round(4),
                         'Recall@K Mean': np.mean(recall).round(4),
                         'F1_Score' : np.mean(f1_score).round(4)}, index=[0])

## Define Cross-Validation
def cross_validation(model, x_train, y_train, k, data, Verbose = True):
    
    kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=28)
    precision_list = []
    recall_list = []
    f1_score_list = []
    roc_list = []
    i=1

    for train_cv, val_cv in kfold.split(x_train, y_train):
        
        if Verbose == True:
            
            print(f'Fold Number {i}/{k}')
            
        else:
            pass
        
        x_train_fold = x_train.iloc[train_cv]
        y_train_fold = y_train.iloc[train_cv]
        x_val_fold = x_train.iloc[val_cv]
        y_val_fold = y_train.iloc[val_cv]

        model_fit = model.fit(x_train_fold, y_train_fold.values.ravel())
        yhat = model.predict_proba(x_val_fold)        
        
        data = x_val_fold.copy()
        data['response'] = y_val_fold.copy()
        data['score'] = yhat[:, 1].tolist()
        data = data.sort_values('score', ascending=False)

        precision = precision_at_k(data) 
        precision_list.append(precision)
        
        recall = recall_at_k(data)
        recall_list.append(recall)
        
        f1_score = round(2*(precision * recall) / (precision + recall), 3)
        f1_score_list.append(f1_score)
       
        roc = roc_auc_score(y_val_fold, yhat[:, 1])
        roc_list.append(roc)
            
        
        i+=1
        
    df = pd.DataFrame({'Model Name': type(model).__name__,
                         'ROC AUC': roc.round(4),
                         'Precision@K Mean': np.mean(precision).round(4),
                         'Recall@K Mean': np.mean(recall).round(4),
                         'F1_Score' : np.mean(f1_score).round(4)}, index=[0])
    return df


# ### 0.1.2 Graphic

# In[3]:


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


# ## 0.2 Loading Data 

# In[4]:


df_raw = pd.read_csv('data/train.csv')


# ## 0.3 Loading Production Data

# In[5]:


df_prod = pd.read_csv('data/test.csv')


# # 1 Data Details

# In[6]:


df1 = df_raw.copy()


# In[7]:


df1.head()


# ## 1.1 Data Dictionary

# |The data set that I am using is from Kaggle (https://www.kaggle.com/anmolkumar/health-insurance-cross-sell-prediction).
# 
# 
# 
# | Feature                                       |Description   
# |:---------------------------|:---------------
# | **Id**                         | Unique ID for the customer   | 
# | **Gender**                           | Gender of the customer   | 
# | **Age**                           | Age of the customer   | 
# | **Driving License**                                   | 0, customer does not have DL; 1, customer already has DL  | 
# | **Region Code**                               | Unique code for the region of the customer   | 
# | **Previously Insured**                     | 1, customer already has vehicle insurance; 0, customer doesn't have vehicle insurance | 
# | **Vehicle Age**                     | Age of the vehicle | 
# | **Vehicle Damage**                                  | 1, customer got his/her vehicle damaged in the past; 0, customer didn't get his/her vehicle damaged in the past | 
# | **Anual Premium**                             | The amount customer needs to pay as premium in the year | 
# | **Policy sales channel**                                    | Anonymized Code for the channel of outreaching to the customer ie  | 
# | **Vintage**                | Number of Days, customer has been associated with the company  | 
# | **Response**              | 1, customer is interested; 0, customer is not interested. |    

# ## 1.2 Rename Columns

# In[8]:


cols_old = ['id', 'Gender', 'Age', 'Driving_License', 'Region_Code', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage', 'Annual_Premium', 
            'Policy_Sales_Channel', 'Vintage', 'Response']

snakecase = lambda x: inflection.underscore( x )

cols_new = list( map( snakecase, cols_old ) )

# rename
df1.columns = cols_new


# ## 1.3 Data Dimensions

# In[9]:


print ('Number of Rows: {}'.format( df1.shape[0]))
print ('Number of Columns: {}'.format( df1.shape[1]))


# ## 1.4 Data Types

# In[10]:


df1.dtypes


# ## 1.5 Missing Values

# In[11]:


df1.isna().sum()


# ## 1.6 Change Types

# In[12]:


# changing data types from float to int64

df1['region_code'] = df1['region_code'].astype('int64')     

df1['policy_sales_channel'] = df1['policy_sales_channel'].astype('int64')    

df1['annual_premium'] = df1['annual_premium'].astype('int64')    


# ## 1.7 Descriptive Statistics

# In[13]:


# Split numerical and categorical features
num_attributes = df1.select_dtypes( include=['int64', 'float64'])
cat_attributes = df1.select_dtypes( exclude=['int64', 'float64'])


# ### 1.7.1 Numerical Attributes

# In[14]:


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

# ### 1.7.2 Categorical Attributes

# In[15]:


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

# In[16]:


categorical_combo = pd.DataFrame(round(cat_attributes.value_counts(normalize=True) * 100)).reset_index().rename(columns={0: '%'})
categorical_combo['count'] = cat_attributes.value_counts().values
display(categorical_combo)


# **1.** Males with car age between 1-2 years old that got vehicle damaged in the past 
# 
# **2.** Female with car newer than 1 year old that never got vehicle damage in the past
# 
# **3.** Males with car newer than 1 year old that never got vehicle damage in the past

# # 2 Feature Engineering

# In[17]:


df2 = df1.copy()


# ## 2.1 Features Creation 

# In[18]:


# vehicle age
df2['vehicle_age']= df2['vehicle_age'].apply( lambda x: 'over_2_years' if x == '> 2 Years' else 'between_1_2_years' if x== '1-2 Year' else 'under_1_year')
# vehicle damage
df2['vehicle_damage'] = df2['vehicle_damage'].apply( lambda x: 1 if x == 'Yes' else 0 )


# ## 2.2 Mind Map

# In[19]:


Image ('img/mindmap.png')


# ## 2.3 Hypothesis Formulation

# ### 2.3.1 User/Client

# **1.** Interest is greater on older people (over 45)
# 
# **2.** Interest is greater on female
# 
# **3.** Interest is greater on customers holding a driving license
# 
# **4.** Interest is greater on customers with higher household numbers
# 
# **5.** Interest is greater on customers with higher annual income
# 
# **6.** Interest is greater on more time associated users (over 150 days)

# ### 2.3.2 Vehicle

# **1.** Interest is greater with newer vehicles
# 
# **2.** Interest is greater with higher historical number of owners
# 
# **3.** Interest is greater with higher modification vehicles
# 
# **4.** Interest is greater with bigger engine size
# 
# **5.** Interest is greater with previously damaged vehicles
# 
# **6.** Interest is greater with higher value vehicles

# ### 2.3.3 Insurance

# **1.** Interest is greater on previously insured owners
# 
# **2.** Interest is greater on phone driven sales
# 
# **3.** Interest is greater on higher annual premiums (over 30k/year)

# ## 2.4 Final List of Hypothesis
# **This comprises the list of hypothesis that can be tested with currently available data**

# **1.** Interest is greater on older people (over 45)
# 
# **2.** Interest is greater on female
# 
# **3.** Interest is greater on customers holding a driving license
# 
# **4.** Interest is greater on more time associated users (over 150 days)
# 
# **5.** Interest is greater with newer vehicles
# 
# **6.** Interest is greater with previously damaged vehicles
# 
# **7.** Interest is greater on previously insured owners
# 
# **8.** Interest is greater on higher annual premiums (over 30k/year)

# # 3 Exploratory Data Analysis (EDA)

# In[20]:


df3 = df2.copy()


# ## 3.1 Univariate Analysis

# ### 3.1.1 Response Variable

# In[21]:


sns.countplot(x = 'response', data=df3);


# ### 3.1.2 Numerical Variables

# In[22]:


num_attributes.hist(bins=25);


# #### 3.1.2.1 Age
# 
# **Findings** The average age of interested clients is higher than non-interested clients. Both plots disclose well how younger clients are not as interested as older clients.

# In[23]:


plt.subplot(1, 2, 1)
sns.boxplot( x='response', y='age', data=df3 )

plt.subplot(1, 2, 2)
sns.histplot(df3, x='age', hue='response');


# #### 3.1.2.2 Driving Licence
# **Findings** Only clients holding a driving license are part of the dataset. 12% are potential vehicle insurance customers

# In[24]:


aux2 = pd.DataFrame(round(df3[['driving_license', 'response']].value_counts(normalize=True) * 100)).reset_index().rename(columns={0: '%'})
aux2['count'] = (aux2['%'] * df3.shape[0]).astype(int)
aux2


# In[25]:


aux2 = df3[['driving_license', 'response']].groupby( 'response' ).sum().reset_index()
sns.barplot( x='response', y='driving_license', data=aux2 );


# #### 3.1.2.3 Region Code

# In[26]:


aux3 = df3[['id', 'region_code', 'response']].groupby( ['region_code', 'response'] ).count().reset_index()
aux3 = aux3[(aux3['id'] > 1000) & (aux3['id'] < 20000)]
sns.scatterplot( x='region_code', y='id', hue='response', data=aux3 );


# #### 3.1.2.4 Previously Insured
# **Findings** All potential vehicle insurance customers have never held an insurance. 46% of our clients already have vehicle insurance and are not interested.

# In[27]:


aux4 = pd.DataFrame(round(df3[['previously_insured', 'response']].value_counts(normalize=True) * 100)).reset_index().rename(columns={0: '%'})
aux4['count'] = (aux4['%'] * df3.shape[0]).astype(int)
aux4


# In[28]:


sns.barplot(data=aux4, x='previously_insured', y='count', hue='response');


# #### 3.1.2.5 Annual Premium
# **Findings** Annual premiums for both interested and non-interested clients are very similar.

# In[29]:


aux5 = df3[(df3['annual_premium'] <100000)]
sns.boxplot( x='response', y='annual_premium', data=aux5 );


# #### 3.1.2.6 Policy Sales Channel

# In[30]:


aux6 = df3[['policy_sales_channel', 'response']].groupby( 'policy_sales_channel' ).sum().reset_index()

plt.xticks(rotation=90)
ax6 = sns.barplot( x='response', y='policy_sales_channel', data=aux6, order = aux6['response']);


# #### 3.1.2.7 Vintage

# In[31]:


plt.subplot(1, 2, 1)
sns.boxplot( x='response', y='vintage', data=df3 )

plt.subplot(1, 2, 2)
sns.histplot(df3, x='vintage', hue='response');


# ### 3.1.3 Categorical Variables

# #### 3.1.3.1 Gender

# In[32]:


aux7=pd.crosstab(df3['gender'], df3['response'])
aux7['percent'] = aux7[1]/(aux7[1]+aux7[0])
aux7


# In[33]:


sns.countplot(x= df3['gender'], hue=df3['response']);


# #### 3.1.3.2 Vehicle Age

# In[34]:


aux8 = pd.DataFrame(round(df3[['vehicle_age', 'response']].value_counts(normalize=True) * 100)).reset_index().rename(columns={0: '%'})
aux8['count'] = (aux8['%'] * df3.shape[0]).astype(int)
aux8


# In[35]:


sns.barplot(data=aux8, x='vehicle_age', y='count', hue='response');


# #### 3.1.3.3 Vehicle Damage

# In[36]:


aux9 = pd.DataFrame(round(df3[['vehicle_damage', 'response']].value_counts(normalize=True) * 100)).reset_index().rename(columns={0: '%'})
aux9['count'] = (aux9['%'] * df3.shape[0]).astype(int)
aux9


# In[37]:


sns.barplot(data=aux9, x='vehicle_damage', y='count', hue='response');


# ## 3.2 Bivariate Analysis

# ### 3.2.1 H1. Interest is greater on older people (over 45)
# **True** Hypothesis

# In[38]:


aux211 = df3[df3['age']>45][['id','response']]
aux212 = df3[df3['age']<=45][['id','response']]

fig, axs = plt.subplots(ncols= 2, figsize = (20,8))
sns.countplot(aux211['response'], ax=axs[0]).set_title('Age > 45')
sns.countplot(aux212['response'], ax=axs[1]).set_title('Age <= 45');


# In[39]:


print('% Interested in vehicle insurance for people + 45 years old: {0:.2f}'.format(100*(aux211[aux211['response']==1]['response'].count()/(aux211[aux211['response']==1]['response'].count()+aux211[aux211['response']==0]['response'].count()))))
print('% Interested in vehicle insurance for people - 45 years old: {0:.2f}'.format(100*(aux212[aux212['response']==1]['response'].count()/(aux212[aux212['response']==1]['response'].count()+aux212[aux212['response']==0]['response'].count()))))


# ### 3.2.2 H2. Interest is greater on female
# **False** Hypothesis

# In[40]:


fig, axs = plt.subplots(figsize = (15,8))
sns.countplot(x= df3['gender'], hue=df3['response']);


# In[41]:


aux22=pd.crosstab(df3['gender'], df3['response'])
aux22['percentage'] = aux22[1]/(aux22[1]+aux22[0])
aux22


# ### 3.2.3 H3. Interest is greater on customers holding a driving license
# **True** Hypothesis

# In[42]:


fig, axs = plt.subplots(figsize = (15,8))
sns.countplot(x= df3['driving_license'], hue=df3['response']);


# In[43]:


aux23=pd.crosstab(df3['driving_license'], df3['response'])
aux23['percentage'] = aux23[1]/(aux23[1]+aux23[0])
aux23


# ### 3.2.4 H4. Interest is greater on more time associated users (over 150 days)
# **False** Hypothesis

# In[44]:


aux241 = df3[df3['vintage']>150][['id','response']]
aux242 = df3[df3['vintage']<=150][['id','response']]

fig, axs = plt.subplots(ncols= 2, figsize = (20,8))
sns.countplot(aux241['response'], ax=axs[0]).set_title('Vintage > 150')
sns.countplot(aux242['response'], ax=axs[1]).set_title('Vintage <= 150');


# In[45]:


print('% Interested in vehicle insurance for customers vintage under 150 days: {0:.2f}'.format(100*(aux242[aux242['response']==1]['response'].count()/(aux242[aux242['response']==1]['response'].count()+aux242[aux242['response']==0]['response'].count()))))
print('% Interested in vehicle insurance for customers vintage over 150 days: {0:.2f}'.format(100*(aux241[aux241['response']==1]['response'].count()/(aux241[aux241['response']==1]['response'].count()+aux241[aux241['response']==0]['response'].count()))))


# ### 3.2.5 Interest is greater with newer vehicles
# **False,** Hypothesis

# In[46]:


aux251 = df3[df3['vehicle_age']=='under_1_year'][['id','response']]
aux252 = df3[df3['vehicle_age']=='between_1_2_years'][['id','response']]
aux253 = df3[df3['vehicle_age']=='over_2_years'][['id','response']]

fig, axs = plt.subplots(ncols= 3, figsize = (20,8))
sns.countplot(aux251['response'], ax=axs[0]).set_title('Vehicle Age < 1 Year')
sns.countplot(aux252['response'], ax=axs[1]).set_title('Vehicle Age Between 1-2 Years')
sns.countplot(aux253['response'], ax=axs[2]).set_title('Vehicle Age > 2 Years');


# In[47]:


print('% Interested in vehicle insurance for customers with cars age under 1 year: {0:.2f}'.format(100*(aux251[aux251['response']==1]['response'].count()/(aux251[aux251['response']==1]['response'].count()+aux251[aux251['response']==0]['response'].count()))))
print('% Interested in vehicle insurance for customers with cars age between 1 and 2 years: {0:.2f}'.format(100*(aux252[aux252['response']==1]['response'].count()/(aux252[aux252['response']==1]['response'].count()+aux252[aux252['response']==0]['response'].count()))))
print('% Interested in vehicle insurance for customers with cars age over 2 years: {0:.2f}'.format(100*(aux253[aux253['response']==1]['response'].count()/(aux253[aux253['response']==1]['response'].count()+aux253[aux253['response']==0]['response'].count()))))


# ### 3.2.6 Interest is greater with previously damaged vehicles
# **True,** Hypothesis

# In[48]:


fig, axs = plt.subplots(figsize = (15,8))
aux26 = sns.countplot(df3['vehicle_damage'], hue=df3['response'])
aux26.set_xticklabels(['Non-Previously Damaged', 'Previously Damaged'])
plt.legend(title='Response', loc='upper right', labels=['Interested', 'Not Interested'])
plt.show(aux26);


# In[49]:


aux26=pd.crosstab(df3['vehicle_damage'], df3['response'])
aux26['percentage'] = aux26[1]/(aux26[1]+aux26[0])
aux26 = aux26.rename(index = {0 : 'Non-Previously Damaged', 1: 'Previously Damaged'})
aux26


# ### 3.2.7 Interest is greater on previously insured owners
# **False,** Hypothesis

# In[50]:


fig, axs = plt.subplots( figsize = (15,8))
sns.countplot(df3['previously_insured'], hue=df3['response']);


# In[51]:


aux27=pd.crosstab(df3['previously_insured'], df3['response'])
aux27['percentage'] = aux27[1]/(aux27[1]+aux27[0])
aux27


# ### 3.2.8 Interest is greater on higher annual premiums (over 30k/year)
# **False,** Hypothesis

# In[52]:


aux281 = df3[df3['annual_premium']>30000][['id','response']]
aux282 = df3[df3['annual_premium']<=30000][['id','response']]

fig, axs = plt.subplots(ncols= 2, figsize = (20,8))
sns.countplot(aux281['response'], ax=axs[0]).set_title('Annual_premium > 30k')
sns.countplot(aux282['response'], ax=axs[1]).set_title('Annual_premium < 30k');


# In[53]:


print('% Interested in vehicle insurance for customers with health insurance premium under 30k/year: {0:.2f}'.format(100*(aux252[aux252['response']==1]['response'].count()/(aux252[aux252['response']==1]['response'].count()+aux252[aux252['response']==0]['response'].count()))))
print('% Interested in vehicle insurance for customers with health insurance premium over 30k/year: {0:.2f}'.format(100*(aux251[aux251['response']==1]['response'].count()/(aux251[aux251['response']==1]['response'].count()+aux251[aux251['response']==0]['response'].count()))))


# ## 3.3 Multivariate Analysis

# ### 3.3.1 Numerical Attributed
# **Finding** Having the target variable in scope, the stronger correlations with feature 'Previously Insured' (-0.34), 'Policy Sales Channel' (-0.14) and 'Age' (0.11). Outside the target variable scope, between Age and Policy Sales Chanel there is strong negative correlation of -0.58), 'Previously Insured' and 'Age' of -0.25 and last between 'Previously Insured' and 'Policy Sales Channel' 0.22. 

# In[54]:


corr_matrix= num_attributes.corr()
# Half matrix
mask = np.zeros_like(corr_matrix)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr_matrix, mask = mask, annot = True, square = True, cmap='YlGnBu');


# # 4 Data Preparation

# In[189]:


df4 = df3.copy()


# ## 4.1 Standardization of DataSets 

# In[190]:


ss_annual_premium = StandardScaler()

#annual_premium
df4['annual_premium'] = ss_annual_premium.fit_transform( df4[['annual_premium']].values)
#save to pickle
pickle.dump(ss_annual_premium, open( '/Users/User/repos/InsuranceCS/parameter/annual_premium_scaler.pkl', 'wb'))


# ## 4.2 Rescaling

# In[191]:


mms_age = MinMaxScaler()
mms_vintage = MinMaxScaler()

#age
df4['age'] = mms_age.fit_transform( df4[['age']].values)
#save to pickle
pickle.dump(mms_age, open( '/Users/User/repos/InsuranceCS/webapp/parameter/age_scaler.pkl', 'wb'))

#vintage
df4['vintage'] = mms_vintage.fit_transform( df4[['vintage']].values)
#save to pickle
pickle.dump(mms_vintage, open( '/Users/User/repos/InsuranceCS/webapp/parameter/vintage_scaler.pkl', 'wb'))


# ## 4.3 Transformation

# ### 4.3.1 Encoding

# In[192]:


#gender - target encoder
target_encode_gender = df4.groupby('gender')['response'].mean()
df4.loc[:, 'gender'] = df4['gender'].map(target_encode_gender)
#save to pickle
pickle.dump(target_encode_gender, open( '/Users/User/repos/InsuranceCS/webapp/parameter/target_encode_gender_scaler.pkl', 'wb'))

# region_code - Target Encoding - as there are plenty of categories (as seen in EDA) it is better not to use one hot encoding and to use 
target_encode_region_code = df4.groupby('region_code')['response'].mean()
df4.loc[:, 'region_code'] = df4['region_code'].map(target_encode_region_code)
#save to pickle
pickle.dump(target_encode_region_code, open( '/Users/User/repos/InsuranceCS/webapp/parameter/target_encode_region_code_scaler.pkl', 'wb'))

#vehicle_age
df4 = pd.get_dummies( df4, prefix='vehicle_age', columns=['vehicle_age'] )

#policy_sales_channel - Frequency encode
fe_policy_sales_channel = df4.groupby('policy_sales_channel').size()/len( df4)
df4['policy_sales_channel'] = df4['policy_sales_channel'].map(fe_policy_sales_channel)
#save to pickle
pickle.dump(fe_policy_sales_channel, open( '/Users/User/repos/InsuranceCS/webapp/parameter/fe_policy_sales_channel_scaler.pkl', 'wb'))


# # 5 Feature Selection

# In[59]:


df5 = df4.copy()


# ## 5.1 Split dataframe into training and test

# In[60]:


X = df5.drop('response', axis=1)
y = df5['response'].copy()

x_train, x_val, y_train, y_val = model_selection.train_test_split(X, y, test_size = 0.2)

df5 = pd.concat ( [x_train, y_train], axis = 1)


# ## 5.2 Feature Importance

# In[61]:


forest = ensemble.ExtraTreesClassifier( n_estimators = 250, random_state = 42, n_jobs = -1)

x_train_n = df5.drop(['id', 'response'], axis=1 )
y_train_n = y_train.values
forest.fit( x_train_n, y_train_n)


# In[62]:


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


# # 6 Machine Learning Modelling

# In[63]:


#I will use as well 'driving_license' as it seemes an importante feature in EDA
cols_selected = ['vintage', 'annual_premium', 'age', 'region_code', 
                 'vehicle_damage', 'policy_sales_channel', 'driving_license']

cols_not_selected = ['previously_insured', 'vehicle_age_between_1_2_years', 'vehicle_age_under_1_year', 'gender', 'vehicle_age_over_2_years']

#create df to be used for business understading
x_validation = x_val.drop(cols_not_selected, axis=1)

#create dfs for modeling
x_train = df5[cols_selected]
x_val = x_val[cols_selected]


# ## 6.1 Logistic Regression

# ### 6.1.1 Model Building

# In[64]:


#define model
lr = linear_model.LogisticRegression (random_state = 42)

#train model
lr.fit( x_train, y_train)

#model prediction
yhat_lr = lr.predict_proba( x_val)


# ### 6.1.2 Model Single Performance

# In[65]:


accuracy_lr = accuracy(lr, x_val, y_val, yhat_lr)
accuracy_lr


# ### 6.1.3 Cross Validation Performance

# In[66]:


accuracy_cv_lr = cross_validation(lr, x_train, y_train, 5, df5, Verbose = True)
accuracy_cv_lr


# ### 6.1.4 Performance Plotted

# In[67]:


# Accumulative Gain
skplt.metrics.plot_cumulative_gain(y_val, yhat_lr, figsize = (10, 5));


# In[68]:


# Lift Curve
skplt.metrics.plot_lift_curve( y_val, yhat_lr, figsize = (10, 5) );


# ## 6.2 Naive Bayes

# ### 6.2.1 Model Building

# In[69]:


#define model
naive = GaussianNB()

#train model
naive.fit( x_train, y_train)

#model prediction
yhat_naive = naive.predict_proba( x_val)


# ### 6.2.2 Model Single Performance

# In[70]:


accuracy_naive = accuracy(naive, x_val, y_val, yhat_naive)
accuracy_naive


# ### 6.2.3 Cross Validation Performance

# In[71]:


accuracy_cv_naive = cross_validation(naive, x_train, y_train, 5, df5, Verbose = True)
accuracy_cv_naive


# ### 6.2.4 Performance Plotted

# In[72]:


# Accumulative Gain
skplt.metrics.plot_cumulative_gain(y_val, yhat_naive, figsize = (10, 5));


# In[73]:


# Lift Curve
skplt.metrics.plot_lift_curve( y_val, yhat_naive, figsize = (10, 5) );


# ## 6.3 Extra Trees

# ### 6.3.1 Model Building

# In[74]:


#define model
et = ensemble.ExtraTreesClassifier (random_state = 42, n_jobs=-1)

#train model
et.fit( x_train, y_train)

#model prediction
yhat_et = et.predict_proba( x_val)


# ### 6.3.2 Model Single Performance

# In[75]:


accuracy_et = accuracy(et, x_val, y_val, yhat_et)
accuracy_et


# ### 6.3.3 Cross Validation Performance

# In[76]:


accuracy_cv_et = cross_validation(et, x_train, y_train, 5, df5, Verbose = True)
accuracy_cv_et


# ### 6.3.4 Performance Plotted

# In[77]:


# Accumulative Gain
skplt.metrics.plot_cumulative_gain(y_val, yhat_et, figsize = (10, 5));


# In[78]:


# Lift Curve
skplt.metrics.plot_lift_curve( y_val, yhat_et, figsize = (10, 5) );


# ## 6.4 Random Forest Regressor

# ### 6.4.1 Model Building

# In[79]:


#define model
rf=RandomForestClassifier(n_estimators=100, min_samples_leaf=25)

#train model
rf.fit( x_train, y_train)

#model prediction
yhat_rf = rf.predict_proba( x_val)


# ### 6.4.2 Model Single Performance

# In[80]:


accuracy_rf = accuracy(rf, x_val, y_val, yhat_rf)
accuracy_rf


# ### 6.4.3 Cross Validation Performance

# In[81]:


accuracy_cv_rf = cross_validation(rf, x_train, y_train, 5, df5, Verbose = True)
accuracy_cv_rf


# ### 6.4.4 Performance Plotted

# In[82]:


# Accumulative Gain
skplt.metrics.plot_cumulative_gain(y_val, yhat_rf, figsize = (10, 5));


# In[83]:


# Lift Curve
skplt.metrics.plot_lift_curve( y_val, yhat_rf, figsize = (10, 5) );


# ## 6.5 KNN Classifier

# ### 6.5.1 Model Building

# In[84]:


#define model
knn = neighbors.KNeighborsClassifier (n_neighbors = 8)

#train model
knn.fit( x_train, y_train)

#model prediction
yhat_knn = knn.predict_proba( x_val)


# ### 6.5.2 Model Single Performance

# In[85]:


accuracy_knn = accuracy(knn, x_val, y_val, yhat_knn)
accuracy_knn


# ### 6.5.3 Cross Validation Performance

# In[86]:


accuracy_cv_knn = cross_validation(knn, x_train, y_train, 5, df5, Verbose = True)
accuracy_cv_knn


# ### 6.5.4 Performance Plotted

# In[87]:


# Accumulative Gain
skplt.metrics.plot_cumulative_gain(y_val, yhat_knn, figsize = (10, 5));


# In[88]:


# Lift Curve
skplt.metrics.plot_lift_curve( y_val, yhat_knn, figsize = (10, 5) );


# ## 6.6 XGBoost Classifier

# ### 6.6.1 Model Building

# In[89]:


#define model
xgboost = XGBClassifier(objective='binary:logistic',
                        eval_metric='error',
                        n_estimators = 100,
                        random_state = 22)

#train model
xgboost.fit( x_train, y_train)

#model prediction
yhat_xgboost = xgboost.predict_proba( x_val)


# ### 6.6.2 Model Single Performance

# In[90]:


accuracy_xgboost = accuracy(xgboost, x_val, y_val, yhat_xgboost)
accuracy_xgboost


# ### 6.6.3 Cross Validation Performance

# In[91]:


accuracy_cv_xgboost = cross_validation(xgboost, x_train, y_train, 5, df5, Verbose = True)
accuracy_cv_xgboost


# ### 6.6.4 Performance Plotted

# In[92]:


# Accumulative Gain
skplt.metrics.plot_cumulative_gain(y_val, yhat_xgboost, figsize = (10, 5));


# In[93]:


# Lift Curve
skplt.metrics.plot_lift_curve( y_val, yhat_xgboost, figsize = (10, 5) );


# ## 6.7 LightGBM Classifier

# ### 6.7.1 Model Building

# In[94]:


#define model
lgbm = LGBMClassifier(random_state = 22)

#train model
lgbm.fit( x_train, y_train)

#model prediction
yhat_lgbm = lgbm.predict_proba( x_val)


# ### 6.7.2 Model Single Performance

# In[95]:


accuracy_lgbm = accuracy(lgbm, x_val, y_val, yhat_lgbm)
accuracy_lgbm


# ### 6.7.3 Cross Validation Performance

# In[96]:


accuracy_cv_lgbm = cross_validation(lgbm, x_train, y_train, 5, df5, Verbose = True)
accuracy_cv_lgbm


# ### 6.7.4 Performance Plotted

# In[97]:


# Accumulative Gain
skplt.metrics.plot_cumulative_gain(y_val, yhat_lgbm, figsize = (10, 5));


# In[98]:


# Lift Curve
skplt.metrics.plot_lift_curve( y_val, yhat_lgbm, figsize = (10, 5) );


# ## 6.8 CatBoost Classifier

# ### 6.8.1 Model Building

# In[99]:


#define model
catboost = CatBoostClassifier(verbose = False, random_state = 22)

#train model
catboost.fit( x_train, y_train)

#model prediction
yhat_catboost = catboost.predict_proba( x_val)


# ### 6.8.2 Model Single Performance

# In[100]:


accuracy_catboost = accuracy(catboost, x_val, y_val, yhat_catboost)
accuracy_catboost


# ### 6.8.3 Cross Validation Performance

# In[101]:


accuracy_cv_catboost = cross_validation(catboost, x_train, y_train, 5, df5, Verbose = True)
accuracy_cv_catboost


# ### 6.8.4 Performance Plotted

# In[102]:


# Accumulative Gain
skplt.metrics.plot_cumulative_gain(y_val, yhat_catboost, figsize = (10, 5));


# In[103]:


# Lift Curve
skplt.metrics.plot_lift_curve(y_val, yhat_catboost, figsize=(10, 5));


# ## 6.9 Comparing Models Performance

# ### 6.9.1 Single Performance

# In[104]:


models_results = pd.concat([accuracy_lr, accuracy_naive, accuracy_et, accuracy_rf, accuracy_knn, accuracy_xgboost, accuracy_lgbm, accuracy_catboost])
models_results.sort_values('Recall@K Mean', ascending = False)


# ### 6.9.2 Cross Validation Performance

# In[105]:


models_results_cv = pd.concat([accuracy_cv_lr, accuracy_cv_naive, accuracy_cv_et, accuracy_cv_rf, accuracy_cv_knn, accuracy_cv_xgboost, accuracy_cv_lgbm, accuracy_cv_catboost])
models_results_cv.sort_values('Recall@K Mean', ascending = False)


# # 7 Hyperparameter Fine Tuning

# ## 7.1 Random Search

# In[108]:


import random

param = {'num_leaves' : [2, 5, 10, 30, 50],
         'max_depth' : [-1, 0, 1, 5, 20],
         'num_iterations': [50, 100],
         'learning_rate' : [0.001, 0.01, 0.1, 0.2, 0.3],
         'random_state' : [22]
        }

MAX_EVAL = 10


# In[109]:


final_result = pd.DataFrame({'ROC AUC': [], 'Precision@K Mean': [], 'Recall@K Mean': [], 'F1_Score': [] })

for i in range ( MAX_EVAL ):
    
    ## choose randomly values for parameters
    hp = { k: random.sample(v, 1)[0] for k, v in param.items() }
    print( 'Step ' +str(i +1) + '/' + str(MAX_EVAL))
    print( hp )
    # model
    model_lgbm = LGBMClassifier(num_leaves=hp['num_leaves'],
                                max_depth = hp['max_depth'],
                                num_iterations = hp['num_iterations'],
                                learning_rate=hp['learning_rate'],
                                random_state=hp['random_state'])
                                      

    # performance
    model_lgbm_result = cross_validation(model_lgbm, x_train, y_train, 5, df5, Verbose = False)
    final_result = pd.concat([final_result, model_lgbm_result])
    
final_result.sort_values('Recall@K Mean', ascending = False)


# ## 7.2 Final Model
# **The best parameters are the standard used by the Classifier**

# In[110]:


#define model
lgbm_tuned = LGBMClassifier(random_state = 22)

#train model
lgbm_tuned = lgbm_tuned.fit( x_train, y_train)

#save model to pickle
pickle.dump(lgbm_tuned, open( '/Users/User/repos/InsuranceCS/webapp/model/model_lgbm.pkl', 'wb'))

#model prediction
yhat_lgbm_tuned = lgbm_tuned.predict_proba( x_val)


# In[111]:


accuracy_lgbm_tuned = cross_validation(lgbm_tuned, x_train, y_train, 5, df5, Verbose = False)
accuracy_lgbm_tuned


# # 8 Final Results
# Final Model applied on the performance (test) dataset

# In[112]:


df_prod.head()


# ## 8.1 Feature Engineering

# In[113]:


## Rename Columns
cols_old = ['id', 'Gender', 'Age', 'Driving_License', 'Region_Code', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage', 'Annual_Premium', 
            'Policy_Sales_Channel', 'Vintage']
snakecase = lambda x: inflection.underscore( x )
cols_new = list( map( snakecase, cols_old ) )
# rename
df_prod.columns = cols_new

## Changing data types from float to int64
df_prod['region_code'] = df_prod['region_code'].astype('int64')     
df_prod['policy_sales_channel'] = df_prod['policy_sales_channel'].astype('int64')    
df_prod['annual_premium'] = df_prod['annual_premium'].astype('int64')

## Create copy to use at the end of the deployment
final = df_prod.copy()

## Features Creation
# vehicle age
df_prod['vehicle_age']= df_prod['vehicle_age'].apply( lambda x: 'over_2_years' if x == '> 2 Years' else 'between_1_2_years' if x== '1-2 Year' else 'under_1_year')
# vehicle damage
df_prod['vehicle_damage'] = df_prod['vehicle_damage'].apply( lambda x: 1 if x == 'Yes' else 0 )


# ## 8.2 Data Preparation

# In[114]:


##Standardization
#annual_premium
df_prod['annual_premium'] = ss_annual_premium.fit_transform( df_prod[['annual_premium']].values)

##Rescaling
#age
df_prod['age'] = mms_age.fit_transform( df_prod[['age']].values)
#vintage
df_prod['vintage'] = mms_vintage.fit_transform( df_prod[['vintage']].values)

##Transformation
#gender - target encoder
df_prod.loc[:, 'gender'] = df_prod['gender'].map(target_encode_gender)
# region_code - Target Encoding - as there are plenty of categories (as seen in EDA) it is better not to use one hot encoding and to use 
df_prod.loc[:, 'region_code'] = df_prod['region_code'].map(target_encode_region_code)
#vehicle_age
df_prod = pd.get_dummies( df_prod, prefix='vehicle_age', columns=['vehicle_age'] )
#policy_sales_channel - Frequency encode
fe_policy_sales_channel = df_prod.groupby('policy_sales_channel').size()/len( df_prod)
df_prod['policy_sales_channel'] = df_prod['policy_sales_channel'].map(fe_policy_sales_channel)

##Select Columns from feature importance selector
df_prod = df_prod[cols_selected]


# ##  8.3 Model Application

# In[115]:


#Apply model to dataset
yhat_prod = lgbm_tuned.predict_proba( df_prod)

#Create column scored based on model
final['score'] = yhat_prod[:,1].tolist()
final=final.sort_values('score',ascending=False)

#Replace score by ranking to show the order in which the customers should be called
final=final.reset_index(drop=True)
final['ranking']=final.index+1
final = final.drop('score', axis=1)


# ## 8.4 Final Dataset
# Users ranked dataset

# In[116]:


final.head()


# # 9 Performance Evaluation and Interpretation

# Based on research, the average Motor Insurance premium per year is 900€

# In[117]:


average_motor_insurance = 900


# ## 9.1 Key findings on interested customers most relevant attributes

# ### 9.1.1 Insight #1:
# **Findings** The average age of interested clients is higher than non-interested clients. Both plots disclose well how younger clients are not as interested as older clients.

# In[118]:


aux9111 = df3[df3['age']>45][['id','response']]
aux9112 = df3[df3['age']<=45][['id','response']]

fig, axs = plt.subplots(ncols= 3, figsize = (25,8))
aux9111 = sns.countplot(aux9111['response'], ax=axs[0])
aux9111.set_title('Age > 45')
aux9111.set_xticklabels(['Not Interested', 'Interested'])

aux9112 = sns.countplot(aux9112['response'], ax=axs[1])
aux9112.set_title('Age <= 45')
aux9112.set_xticklabels(['Not Interested', 'Interested'])

aux9113 = sns.histplot(df3, x='age', hue='response')
aux9113.legend(title='Response', loc='upper right', labels=['Interested', 'Not Interested'])
aux9113.set_title('All ages');


# In[119]:


print('% Interested in vehicle insurance for people + 45 years old: {0:.2f}'.format(100*(aux211[aux211['response']==1]['response'].count()/(aux211[aux211['response']==1]['response'].count()+aux211[aux211['response']==0]['response'].count()))))
print('% Interested in vehicle insurance for people - 45 years old: {0:.2f}'.format(100*(aux212[aux212['response']==1]['response'].count()/(aux212[aux212['response']==1]['response'].count()+aux212[aux212['response']==0]['response'].count()))))


# ### 9.1.2 Insight #2:
# **Findings** Interest is greater on customers' with vehicles age over 2 years old.

# In[120]:


aux9121 = df3[df3['vehicle_age']=='under_1_year'][['id','response']]
aux9122 = df3[df3['vehicle_age']=='between_1_2_years'][['id','response']]
aux9123 = df3[df3['vehicle_age']=='over_2_years'][['id','response']]

fig, axs = plt.subplots(ncols= 3, figsize = (20,8))
sns.countplot(aux9121['response'], ax=axs[0]).set_title('Vehicle Age < 1 Year')
sns.countplot(aux9122['response'], ax=axs[1]).set_title('Vehicle Age Between 1-2 Years')
sns.countplot(aux9123['response'], ax=axs[2]).set_title('Vehicle Age > 2 Years');


# In[176]:


print('% Interested in vehicle insurance for customers with cars age under 1 year: {0:.2f}'.format(100*(aux9121[aux9121['response']==1]['response'].count()/(aux9121[aux9121['response']==1]['response'].count()+aux9121[aux9121['response']==0]['response'].count()))))
print('% Interested in vehicle insurance for customers with cars age between 1 and 2 years: {0:.2f}'.format(100*(aux9122[aux9122['response']==1]['response'].count()/(aux9122[aux9122['response']==1]['response'].count()+aux9122[aux9122['response']==0]['response'].count()))))
print('% Interested in vehicle insurance for customers with cars age over 2 years: {0:.2f}'.format(100*(aux9123[aux9123['response']==1]['response'].count()/(aux9123[aux9123['response']==1]['response'].count()+aux9123[aux9123['response']==0]['response'].count()))))


# ### 9.1.3 Insight #3:
# **Findings** Interest is greater on customers' that had his/her vehicle damaged in the past.

# In[122]:


fig, axs = plt.subplots(figsize = (15,8))
aux913 = sns.countplot(df3['vehicle_damage'], hue=df3['response'])
aux913.set_xticklabels(['Non-Previously Damaged', 'Previously Damaged'])
plt.legend(title='Response', loc='upper right', labels=['Interested', 'Not Interested'])
plt.show(aux913);


# In[123]:


aux913=pd.crosstab(df4['vehicle_damage'], df4['response'])
aux913['Percentage'] = round((aux913[1]/(aux913[1]+aux913[0])*100),2)
aux913 = aux913.rename(columns={0: 'Not Interested', 1: 'Interested'})
aux913 = aux913.rename(index = {0 : 'Non-Previously Damaged', 1: 'Previously Damaged'})
aux913


# ### 9.1.4 Insight #4:
# **Findings** Customers with higher health insurance annual Premiums are more interested in getting a vehicle insurance.

# In[178]:


aux9141 = df3[df3['annual_premium']>30000][['id','response']]
aux9142 = df3[df3['annual_premium']<=30000][['id','response']]

fig, axs = plt.subplots(ncols= 2, figsize = (20,8))
aux91411 = sns.countplot(aux9141['response'], ax=axs[0])
aux91411.set_title('Annual_premium > 30k')
aux91411.set_xticklabels(['Not Interested', 'Interested'])

aux91421 = sns.countplot(aux9142['response'], ax=axs[1])
aux91411.set_title('Annual_premium < 30k')
aux91421.set_xticklabels(['Not Interested', 'Interested']);


# In[179]:


print('% Interested in vehicle insurance for customers with health insurance premium under 30k/year: {0:.2f}'.format(100*(aux9142[aux9142['response']==1]['response'].count()/(aux9142[aux9142['response']==1]['response'].count()+aux9142[aux9142['response']==0]['response'].count()))))
print('% Interested in vehicle insurance for customers with health insurance premium over 30k/year: {0:.2f}'.format(100*(aux9141[aux9141['response']==1]['response'].count()/(aux9141[aux9141['response']==1]['response'].count()+aux9141[aux9141['response']==0]['response'].count()))))


# ## 9.2 What percentage of interested customers the sales team will be able to contact making 20.000 calls?

# In[126]:


# Tranform 20.000 to equivalent in Production Dataset - the plots use percentages and not nominal values, therefore, this transformation will be done to further use in data visualization
percent_20000 = int(round(((20000 / df_prod.shape[0]) * x_val.shape[0]),0))
percent_20000


# In[127]:


# Define dataset to apply final model
df9 = x_val.copy()
df9['response'] = y_val.copy()
df9['score'] = yhat_lgbm_tuned[:, 1].tolist()
df9 = df9.sort_values('score', ascending=False)

# Define total number of interested in the dataset
interested = df9[df9['response']==1]
interested['response'].count()

# Define dataset percentage that defines 20.000 calls
percent = round((percent_20000/len(df9)*100), 2)
# Apply Recall metric to define the percentage of interested customers was achieved by phone calling 20.000 people
recall_at_20000 = round((recall_at_k(df9, 12000))*100 , 2 )

print(f'Using {percent}% of the test data, the model could find {recall_at_20000}% of the total customers interested in purchase a car insurance.')


# In[128]:


# Define nominal value of interested customers found in the baseline and in the the tuned model
interested_baseline20k = int(round(((len(interested)*percent))/100,0))
interested_lgbm20k = int(round(((len(interested)*recall_at_20000))/100,0))


# In[169]:


fig, ax = plt.subplots(figsize = (25, 10))

plt.subplot(1, 2, 1)
plot_cumulative_gain(y_val, yhat_lgbm_tuned[:,1])
plt.axvline(1.574, color ='#FF7F0E', ls = '--')
plt.title('Cumulative Gain Final Model - 20.000 calls')
plt.yticks(np.arange(0, 100, step = 10))
plt.xticks(np.arange(0, 10.5, step = 0.5))
plt.legend(['Final Model - LGBM Classifier', 'Wizard', 'Random', 'Test Percentage'])

# display message
message = f'With 20.000 calls: \nRandom Model finds: {(interested_baseline20k):,} new customers\nFinal Model finds:       {(interested_lgbm20k):,} new customers (+{(interested_lgbm20k - interested_baseline20k):,})\n \nGrowth in Revenue of {((interested_lgbm20k - interested_baseline20k) * average_motor_insurance):,}€'
plt.text( 5.5 , 18, message, color = 'black', fontsize = 'medium' )

bbox = dict(boxstyle ='round', fc ='0.7')
arrowprops = dict(facecolor ='#4D4D4D')
plt.annotate('Final Model: 45.44%', xy = (1.58, 45.6),
                xytext =(2.5, 48), 
                arrowprops = arrowprops, bbox = bbox)

plt.annotate('Random: ~15.7%', xy = (1.58, 15.9),
                xytext =(2.2, 18), 
                arrowprops = arrowprops, bbox = bbox)

plt.subplot(1, 2, 2)
plot_lift(y_val, yhat_lgbm_tuned[:,1])
plt.axvline(1.574, color ='#FF7F0E', ls = '--')
plt.title('Lift Curve - 20.000 calls')
plt.yticks(np.arange(0, 3, step = 0.2))
plt.xticks(np.arange(0, 10.5, step = 0.5))
plt.legend(['Final Model - LGBM Classifier', 'Random', 'Test Percentage']);


# ## 9.3 By increasing the capacity to 40,000 calls, what percentage of interested customers the sales team will be able to contact?

# In[139]:


# Tranform 20.000 to equivalent in Production Dataset - the plots use percentages and not nominal values, therefore, this transformation will be done to further use in data visualization
percent_40000 = int(round(((40000 / df_prod.shape[0]) * x_val.shape[0]),0))
percent_40000


# In[140]:


# Define dataset percentage that defines 40.000 calls
percent = round((percent_40000/len(df9)*100), 2)
# Apply Recall metric to define the percentage of interested customers was achieved by phone calling 20.000 people
recall_at_40000 = round((recall_at_k(df9, 24000))*100 , 2 )

print(f'Using {percent}% of the test data, the model could find {recall_at_40000}% of the total customers interested in purchase a car insurance.')


# In[141]:


# Define nominal value of interested customers found in the baseline and in the the tuned model
interested_baseline40k = int(round(((len(interested)*percent))/100,0))
interested_lgbm40k = int(round(((len(interested)*recall_at_40000))/100,0))


# In[157]:


fig, ax = plt.subplots(figsize = (25, 10))

plt.subplot(1, 2, 1)
plot_cumulative_gain(y_val, yhat_lgbm_tuned[:,1])
plt.axvline(3.149, color ='#FF7F0E', ls = '--')
plt.title('Cumulative Gain Final Model - 40.000 calls')
plt.yticks(np.arange(0, 100, step = 10))
plt.xticks(np.arange(0, 10.5, step = 0.5))
plt.legend(['Final Model - LGBM Classifier', 'Wizard', 'Random', 'Test Percentage'])

# display message
message = f'With 40.000 calls: \nRandom Model finds: {(interested_baseline40k):,} new customers\nFinal Model finds:       {(interested_lgbm40k):,} new customers (+{(interested_lgbm40k - interested_baseline40k):,})\n \nGrowth in Revenue of {((interested_lgbm40k - interested_baseline40k) * average_motor_insurance):,}€'
plt.text( 5.5 , 18, message, color = 'black', fontsize = 'medium' )

arrowprops = dict(facecolor ='#4D4D4D')
plt.annotate('Final Model: 78.8%', xy = (3.149, 78.84),
                xytext =(3.8, 73), 
                arrowprops = arrowprops, bbox = bbox)

plt.annotate('Random: ~31.5%', xy = (3.15, 31.49),
                xytext =(3.5, 25), 
                arrowprops = arrowprops, bbox = bbox);



plt.subplot(1, 2, 2)
plot_lift(y_val, yhat_lgbm_tuned[:,1])
plt.axvline(3.149, color ='#FF7F0E', ls = '--')
plt.title('Lift Curve - 40.000 calls')
plt.yticks(np.arange(0, 3, step = 0.2))
plt.xticks(np.arange(0, 10.5, step = 0.5))
plt.legend(['Final Model - LGBM Classifier', 'Random', 'Test Percentage']);

