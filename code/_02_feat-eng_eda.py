# This script performs feature engineering and secondary EDA

# Load Libraries and Change WD======================================================================
## Load libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
import itertools
from sklearn.metrics import matthews_corrcoef
from scipy.stats import chi2_contingency


## Change wd
from pathlib import Path
import os
Path.cwd()
root = '/Users/keithpost/Documents/Python/Python projects/mercari_price_ml_py/'
os.chdir(root + 'code')
Path.cwd()



# Source Functions and Data=========================================================================
## Functions
from _00_helper_objs_fns import (
  fixed_colors, 
  make_countplot, 
  make_histplot, 
  make_barplot,
  map_avg_price,
  log_transform,
  correlation_ratio
)


## Data
os.chdir(root + 'data')
df = pd.read_pickle("train_clean.pkl")



# EDA & New Features================================================================================
## Binary categorical fields (known vs unknown)
### Category name
df['has_cat_name'] = df['category_name'] != 'Unknown/Unknown/Unknown'

### Brand name
df['has_brand'] = df['brand_name'] != 'Unknown'

### Item description
df['has_desc'] = df['item_description'] != 'No item description'


### Univariate plots (counts)
labs_has = ['No', 'Yes']
fig, axes = plt.subplots(1, 3)

make_countplot(df=df, var='has_cat_name', xlabs=labs_has, ylim=1500000, ax=axes[0])
make_countplot(df=df, var='has_brand', xlabs=labs_has, ylim=1500000, lableft=False, 
               ax=axes[1])
make_countplot(df=df, var='has_desc', xlabs=labs_has, ylim=1500000, lableft=False,
               ax=axes[2])
               
fig.suptitle('Numbers of items by whether category name, brand, and description are present',
             fontsize=11)
fig.supylabel('No. of items', fontsize=10)
plt.show()
plt.close()

#all 3 have considerable differences


### Bivariate plots
## Price
fig, axes = plt.subplots(1, 3)

make_barplot(df=df, var='has_cat_name', y='price', xlabs=labs_has, alpha=0.7, ax=axes[0])
make_barplot(df=df, var='has_brand', y='price', xlabs=labs_has, alpha=0.7, ax=axes[1])
make_barplot(df=df, var='has_desc', y='price', xlabs=labs_has, alpha=0.7, ax=axes[2])

fig.suptitle('Average prices by whether category name, brand, and description are present',
             fontsize=11)
fig.supylabel('Mean Price ($) (\u00B1 1 SE)', fontsize=10)

plt.show()
plt.close()

#all 3 vary by price too


## Log price
fig, axes = plt.subplots(1, 3)

make_barplot(df=df, var='has_cat_name', y='price_log', xlabs=labs_has, alpha=0.7, ax=axes[0])
make_barplot(df=df, var='has_brand', y='price_log', xlabs=labs_has, alpha=0.7, ax=axes[1])
make_barplot(df=df, var='has_desc', y='price_log', xlabs=labs_has, alpha=0.7, ax=axes[2])

fig.suptitle('Average prices by whether category name, brand, and description are present',
             fontsize=11)
fig.supylabel('Mean Log-Transformed Price ($) (± 1 SE)', fontsize=10)

plt.show()
plt.close()


## Matching keywords in item_description
keywords = ['new', 'brand new', 'used', 'worn', 'never used', 'authentic']
df['has_keyword_new'] = df['item_description'].str.contains('new', case=False, na=False)
labs_new = ['No', 'Yes']


### Univariate plot
ax = sns.countplot(data=df, x='has_keyword_new', color=fixed_colors['has_keyword_new'])
ax.set_xticklabels(labs_new)
ax.set_xlabel("Has keyword 'new'?")
ax.set_ylabel("No. of items")
ax.set_ylim(0, 1200000)
ax.set_title("Number of items by whether description has the keyword 'new'")

plt.show()
plt.close()


### Bivariate plots
fig, axes = plt.subplots(1, 2)

make_barplot(df=df, var='has_keyword_new', y='price', xlabs=labs_new, 
             ytitle='Mean Price ($) (\u00B1 1 SE)', ax=axes[0])
make_barplot(df=df, var='has_keyword_new', y='price_log', xlabs=labs_new, 
             ytitle='Mean Log-Transformed Price ($) (± 1 SE)',ax=axes[1])
               
fig.suptitle("Average prices (raw and log-transformed) by whether item has keyword 'new'",
             fontsize=11)
             
plt.show()
plt.close()


## Additional features: Price per top 5 of each category
for col in ['dpt_top5', 'cat_top5', 'class_top5', 'brand_top5', 'item_condition_id']:
  map_avg_price(df, col)
#NOTE: will calculate these per fold during CV but will use to assess functional redundancy



# Feature Scaling===================================================================================
## Make qqplots of numerical variables: 
nums = ['name_wc', 'desc_len']
df_nums = df[nums]

### Create grid of subplots
fig, axes = plt.subplots(1, 2)

qqplot_name_wc = sm.qqplot(df['name_wc'], line='s', ax=axes[0])
axes[0].set_title('Name (word count)')
qqplot_name_wc = sm.qqplot(df['desc_len'], line='s', ax=axes[1])
axes[1].set_title('Item description (length)')

plt.tight_layout()
plt.show()
plt.close()
#clear issues with normality

## Transform using Min-max and logs
### Min-max
X_train = df[nums].to_numpy()
scaler= MinMaxScaler().fit(X_train)
scaler
X_scaled_minmax = scaler.transform(X_train)

nums_minmax = [col + '_minmax' for col in nums]
df_nums[nums_minmax] = pd.DataFrame(X_scaled_minmax)


### Log-transform
log_transformer = FunctionTransformer(func=log_transform, inverse_func=np.expm1)
X_scaled_log = log_transformer.fit_transform(X_train)

nums_log = [col + '_log' for col in nums]
df_nums[nums_log] = pd.DataFrame(X_scaled_log)


## Plot results
fig, axes = plt.subplots(2, 2)

qqplot_name_wc = sm.qqplot(df_nums['name_wc_minmax'], line='s', ax=axes[0, 0])
axes[0,0].set_title('Name (min-max scaled word count)')
qqplot_name_wc = sm.qqplot(df_nums['desc_len_minmax'], line='s', ax=axes[0, 1])
axes[0,1].set_title('Item description (min-max scaled length)')
qqplot_name_wc = sm.qqplot(df_nums['name_wc_log'], line='s', ax=axes[1, 0])
axes[1,0].set_title('Name (log-transformed word count)')
qqplot_name_wc = sm.qqplot(df_nums['desc_len_log'], line='s', ax=axes[1, 1])
axes[1,1].set_title('Item description (log-transformed length)')


plt.tight_layout()
plt.show()
plt.close()
#--> for workflow, use MM scaler for name (wc) and log-transform for desc (len)



# Final Checks======================================================================================
# 1. Missing values
#do any features contain missing values? 
df.isnull().sum() #all 0s
#Conclusion: no missing values


# 2. Feature redundancy
## Literal redundancy (test with subset of data)
df_sub = df.sample(5000)
duplicate_columns = df_sub.T.duplicated(keep=False)
redundant_cols = duplicate_columns[duplicate_columns].index.tolist()
len(redundant_cols) #0
#Conclusion: no redundant columns/features


## Functional redundancy or strong collinearity
### Remove extraneous fields (28 to start)
cols_to_drop = ['train_id', 'name', 'category_name', 'brand_name', 'price', 'item_description',
                'department', 'category', 'class', 'desc_wc', 'name_len', 'price_log']
              
df_dropped = df.drop(cols_to_drop, axis=1)
df_dropped.info() #17


### Split data into separate types
#numeric (7)
cols_num = ['dpt_top5_lprice', 'cat_top5_lprice', 'class_top5_lprice', 'brand_top5_lprice',
            'item_condition_id_lprice', 'name_wc', 'desc_len']
df_num = df[cols_num]

#binary (5)
cols_bi = ['shipping', 'has_cat_name', 'has_brand', 'has_desc', 'has_keyword_new']
df_bi = df[cols_bi].astype(int) #convert Boolean to integer type

#categorical (4)
cols_cat = ['dpt_top5', 'cat_top5', 'class_top5', 'brand_top5']
df_cat = df[cols_cat]

#ordinal (1)
cols_ord = ['item_condition_id']
df_ord = df[cols_ord]


### Numeric-numeric: Spearman correlation
#run correlation
df_corr = df_num.corr('spearman').abs() 

#test for # values per column = 1 (expect 1, diagonal)
df_corr.apply(lambda x: sum(x == 1), axis=1).unique() #all 1s

#replace 1s with 0s
df_corr_replace = df_corr.replace(1, 0)

#assess for strong pairwise correlations
df_corr_replace.apply(lambda x: sum(x > 0.9), axis=1).unique() 
#Result: all 0s, so none


#### Binary-binary: Phi coefficient (equivalent to Pearson)
#create empty corr matrix
cols_bi = df_bi.columns
phi_matrix = pd.DataFrame(np.eye(len(cols_bi)), index=cols_bi, columns=cols_bi)

#compute pairwise MCC (phi)
for col1, col2 in itertools.combinations(cols_bi, 2):
    phi = matthews_corrcoef(df_bi[col1], df_bi[col2])
    phi_matrix.loc[col1, col2] = phi
    phi_matrix.loc[col2, col1] = phi  

#convert to abs
phi_matrix = phi_matrix.abs()

#test for # values per column = 1 (expect 1, diagonal)
phi_matrix.apply(lambda x: sum(x == 1), axis=1).unique() #all 1s

#replace 1s with 0s
phi_matrix_replace = phi_matrix.replace(1, 0)

#assess for strong pairwise correlations
phi_matrix_replace.apply(lambda x: sum(x > 0.8), axis=1).unique() 
#Result: all 0s, so none


### Ordinal-ordinal: Spearman correlation
#only one ordinal feature (so no pairs to assess)


### Numeric-binary: Pearson correlation
df_num_bi = None #start with empty DF

#iterate .corrwith to calculate Pearson corrs
for col in df_bi.columns:
  df_corr = df_num.corrwith(df_bi[col]).to_frame(col).abs()
  df_num_bi = pd.concat([df_num_bi, df_corr], axis=1)

df_num_bi

#test for # values per column = 1
df_num_bi.apply(lambda x: sum(x == 1), axis=1).unique() #all 0s, so none

#assess for strong pairwise correlations
df_num_bi.apply(lambda x: sum(x > 0.9), axis=1).unique() 
#Result: all 0s, so none


### Numeric-categorical: correlation ratio
#combine DFs
df_cat_num = pd.concat([df_cat, df_num], axis=1)

#build DF of corr_ratios (start with empty DF with index and col names)
df_corr_ratio = pd.DataFrame(index=cols_cat,
                             columns=cols_num)

#iterate through index and cols to populate DF
for cat in cols_cat:
  for num in cols_num:
    corr_ratio = correlation_ratio(df_cat_num[cat], df_cat_num[num])
    df_corr_ratio.loc[cat, num] = corr_ratio

df_corr_ratio


#test for # values per column near 1
df_corr_ratio.apply(lambda x: sum(x > 0.999999), axis=1).unique() 
#all 1s (expected); because the log-price per top 5 cat are deterministic, they will be retained

#assess for strong pairwise correlations
df_corr_ratio.apply(lambda x: sum(x >= 0.5), axis=1)
#3 (2 more than expected) for cat_top5 and 1 for others

df_corr_ratio.loc['cat_top5']
#Result: cat_top5 is strongly associated with dpt_top5_lprice and class_top5_lprice
#--> because of this and the 1s (expected), this will be dropped for linear models


### Categorical-categorical and categorical-binary: Cramer's V
#combine DFs
df_cat_bi = pd.concat([df_cat, df_bi], axis=1)

#create empty matrix
cols_cat_bi = df_cat_bi.columns
cramers_cat_bi_matrix = pd.DataFrame(np.eye(len(cols_cat_bi)), index=cols_cat_bi, 
                                     columns=cols_cat_bi)

#compute pairwise Cramer's V
for col1, col2 in itertools.combinations(cols_cat_bi, 2):
    #build a contingency table
    confusion_matrix = pd.crosstab(df_cat_bi[col1], df_cat_bi[col2])
    #computes the Chi-squared stat
    chi2 = chi2_contingency(confusion_matrix)[0]
    #n = number of obs
    n = confusion_matrix.sum().sum()
    #k = smaller dim of the contingency table
    k = min(confusion_matrix.shape)
    cramers_v = np.sqrt(chi2/(n*(k-1)))
    
    cramers_cat_bi_matrix.loc[col1, col2] = cramers_v
    cramers_cat_bi_matrix.loc[col2, col1] = cramers_v

cramers_cat_bi_matrix

#test for # values per column = 1 (expect 1, diagonal)
cramers_cat_bi_matrix.apply(lambda x: sum(x == 1), axis=1).unique() #all 1s

#replace 1s with 0s
cramers_cat_bi_matrix_replace = cramers_cat_bi_matrix.replace(1, 0)

#assess for strong pairwise correlations
cramers_cat_bi_matrix_replace.apply(lambda x: sum(x > 0.9), axis=1).unique() 
#Result: all 0s, so none



### Ordinal-numeric: Spearman correlation
df_ord_num_corr = (
  df_num.corrwith(df_ord['item_condition_id'], 
                  method='spearman').to_frame('item_condition_id')
                                    .abs()
  )
len(df_ord_num_corr[df_ord_num_corr['item_condition_id']>0.9])
#Result: 0, so none is greater than 0.9


#4. Categorical variables
#with few categories (after rare label encoding), use one-hot encoding for both types



# Feature Selection=================================================================================
## Remove extraneous features (retain in preferred order)
cols_mod_full = ['train_id', #id
                 'item_condition_id', #ordinal
                 'shipping', 'has_cat_name', 'has_brand', 'has_desc', 'has_keyword_new', #Bi/Bool
                 'name_wc', 'desc_len', #num
                 'department', 'category', 'class', 'brand_name', #categorical
                 'price', 'price_log'] #target

df_mod_full=df[cols_mod_full]


## Convert Boolean cols to integers
cols_bool = ['has_cat_name', 'has_brand', 'has_desc', 'has_keyword_new']
df_mod_full[cols_bool] = df_mod_full[cols_bool].astype(int)



# Save DF===========================================================================================
pd.to_pickle(df_mod_full, "train_clean_feat.pkl")

