#ARCHIVED: script to check pairwise collinearity of features as part of feature selection. No 
  #longer necessary since LM was not considered a prospective model. Features may have been dropped
  #for ridge regression to reduce computational demand

# Load Libraries and Change WD======================================================================
## Load libraries
import pandas as pd              
import numpy as np                
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
from _00_helper_objs_fns import correlation_ratio


## Data
os.chdir(root + 'data')
df = pd.read_pickle("train_collinear_test.pkl")



# Collinearity Testing==============================================================================
## Remove extraneous fields for assessing collinearity 
cols_to_retain = ['item_condition_id', 'shipping', 'name_wc', 'desc_len', 'dpt_top5',
                  'cat_top5', 'class_top5', 'brand_top5', 'has_cat_name', 'has_brand',
                  'has_desc', 'has_keyword_new', 'dpt_top5_lprice', 'cat_top5_lprice',
                  'class_top5_lprice', 'brand_top5_lprice', 'item_condition_id_lprice']
# cols_to_drop = ['train_id', 'name', 'category_name', 'brand_name', 'price', 'item_description',
#                 'department', 'category', 'class', 'desc_wc', 'name_len', 'price_log']
              
df_retained = df[cols_to_retain]
df_retained.info()
# df_dropped = df.drop(cols_to_drop, axis=1)
# df_dropped.info() #17


## Split data into separate types
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


## Numeric-numeric: Spearman correlation
#run correlation
df_corr = df_num.corr('spearman').abs() 

#test for # values per column = 1 (expect 1, diagonal)
df_corr.apply(lambda x: sum(x == 1), axis=1).unique() #all 1s

#replace 1s with 0s
df_corr_replace = df_corr.replace(1, 0)

#assess for strong pairwise correlations
df_corr_replace.apply(lambda x: sum(x > 0.9), axis=1).unique() 
#Result: all 0s, so none


### Binary-binary: Phi coefficient (equivalent to Pearson)
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


## Ordinal-ordinal: Spearman correlation
#only one ordinal feature (so no pairs to assess)


## Numeric-binary: Pearson correlation
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


## Numeric-categorical: correlation ratio
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


## Categorical-categorical and categorical-binary: Cramer's V
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


## Ordinal-numeric: Spearman correlation
df_ord_num_corr = (
  df_num.corrwith(df_ord['item_condition_id'], 
                  method='spearman').to_frame('item_condition_id')
                                    .abs()
  )
len(df_ord_num_corr[df_ord_num_corr['item_condition_id']>0.9])
#Result: 0, so none is greater than 0.9
