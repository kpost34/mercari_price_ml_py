# This script performs feature engineering and secondary EDA

# Load Libraries, Data, and Functions===============================================================
## Load libraries
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import FunctionTransformer


## Data
ROOT = Path.cwd()
data_path_in = ROOT / "data" / "train_clean.pkl"
df = pd.read_pickle(data_path_in)


## Functions
from code._00_helper_objs_fns import (
  fixed_colors, 
  make_countplot, 
  make_histplot, 
  make_barplot,
  map_avg_price,
  log_transform,
  create_boolean_desc_features
)



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
### Create Boolean fields 
df = create_boolean_desc_features(df)
df['has_keyword_new'] = df['item_description'].str.contains('new', case=False, na=False)


### Assess how they differ
#means
df.groupby('has_keyword_new_like', as_index=False)['price_log'].mean()
df.groupby('has_keyword_used_like', as_index=False)['price_log'].mean()
df.groupby('has_keyword_authentic_like', as_index=False)['price_log'].mean()
df.groupby('has_keyword_rare_like', as_index=False)['price_log'].mean()
df.groupby('has_keyword_set_like', as_index=False)['price_log'].mean()

df.groupby('has_keyword_new', as_index=False)['price_log'].mean()


### Univariate plots
#new
labs_new = ['No', 'Yes']
ax = sns.countplot(data=df, x='has_keyword_new', color=fixed_colors['has_keyword_new'])
ax.set_xticklabels(labs_new)
ax.set_xlabel("Has keyword 'new'?")
ax.set_ylabel("No. of items")
ax.set_ylim(0, 1200000)
ax.set_title("Number of items by whether description has the keyword 'new'")

plt.show()
plt.close()


### Bivariate plots
#new
fig, axes = plt.subplots(1, 2)

make_barplot(df=df, var='has_keyword_new', y='price', xlabs=labs_new, 
             ytitle='Mean Price ($) (\u00B1 1 SE)', ax=axes[0])
make_barplot(df=df, var='has_keyword_new', y='price_log', xlabs=labs_new, 
             ytitle='Mean Log-Transformed Price ($) (± 1 SE)',ax=axes[1])
               
fig.suptitle("Average prices (raw and log-transformed) by whether item has keyword 'new'",
             fontsize=11)
             
plt.show()
plt.close()

#authentic-like
fig, axes = plt.subplots(1, 2)

make_barplot(df=df, var='has_keyword_authentic_like', y='price', xlabs=labs_new, 
             ytitle='Mean Price ($) (\u00B1 1 SE)', xtitle='', alpha=0.7, ax=axes[0])
make_barplot(df=df, var='has_keyword_authentic_like', y='price_log', xlabs=labs_new, 
             ytitle='Mean Log-Transformed Price ($) (± 1 SE)', xtitle='', alpha=0.7, ax=axes[1])
               
fig.suptitle("Average prices (raw and log-transformed) by whether item has keywords related to 'authentic'",
             fontsize=11)
fig.supxlabel("Description contains keywords related to 'authentic'", fontsize=9)
    
plt.tight_layout()         
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
# data_path_out_archive = ROOT / "data" / "train_collinear_test.pkl"
# pd.to_pickle(df, data_path_out_archive)
# see archive for details on some pairwise correlational tests
#note: ultimately did not weight heavily because LM was dropped as a prospective model which is
  #greatly affected by collinearity and with TF-IDF would require an iterative drop-refit process
  


#4. Categorical variables
#with few categories (after rare label encoding), use one-hot encoding for both types



# Feature Selection=================================================================================
## Remove extraneous features (retain in preferred order)
cols_mod_full = ['train_id', #id
                 'name', 'item_description', #string (for tf-idf)
                 'item_condition_id', #ordinal
                 'shipping', 'has_cat_name', 'has_brand', 'has_desc', #'has_keyword_new', #Bi/Bool
                 'has_keyword_new_like', 'has_keyword_authentic_like', 'has_keyword_used_like',
                 'has_keyword_rare_like', 'has_keyword_set_like',
                 'name_wc', 'desc_len', #num
                 'department', 'category', 'class', 'brand_name', #categorical
                 'price', 'price_log'] #target

df_mod_full=df[cols_mod_full]


## Convert Boolean cols to integers
cols_bool = ['has_cat_name', 'has_brand', 'has_desc', 'has_keyword_new_like',
'has_keyword_authentic_like', 'has_keyword_used_like', 'has_keyword_rare_like', 
'has_keyword_set_like']
df_mod_full[cols_bool] = df_mod_full[cols_bool].astype(int)



# Save DF===========================================================================================
data_path_out = ROOT / "data" / "train_clean_feat.pkl"
# pd.to_pickle(df_mod_full, data_path_out)

