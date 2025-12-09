# This script does initial data cleaning/wrangling and performs EDA


# Load Libraries and Change WD======================================================================
## Load libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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
  get_top_cat, 
  add_top_cat, 
  make_countplot, 
  make_histplot, 
  make_barplot
)

## Data
os.chdir(root + 'data')
df0 = pd.read_csv("train.tsv", sep='\t')



# Data Checking=====================================================================================
## General info
df0.info() 
#1,482,535 rows x 8 columns
#train_id (int), name (obj), item_condition_id (int), category_name (obj), brand_name(obj),
  #price (float), shipping (int), and item_description (obj)
#category_name, brand_name, and item_description have missing values
  
df0.describe()
df0.head()
df0.index #numerical index from 0 - 1482535
df0.dtypes


## Missingness
### By column
print(df0.isnull().sum())
#none in train_id (expected), name (expected), item_condition_id, price (expected), and shipping
#category_name: 6327
#brand_name: 632682
#item_description: 6

len(df0[df0['item_description']=='No description yet'])
#82489 that have item_description values of 'No description yet' (type of unknown/missing value)


### By row (true nulls)
df0.dropna() #846,981 without NAs
df0.dropna(thresh=1) #1,482,535 with 0-1 NAs (all of them)
#rows are either complete or have 1 NA (none with 2+)
#specifically, rows are complete or have category_name, brand_name, or item_description missing


## Duplicates
print(df0.duplicated().sum()) #no duplicate rows



# Initial Wrangling=================================================================================
## Copy DF
df1 = df0.copy()


## Address missing values
### Item description
df1['item_description'] = df1['item_description'].fillna('No item description')
df1['item_description'] = df1['item_description'].replace('No description yet', 'No item description')


### Brand name
df1['brand_name'] = df1['brand_name'].fillna('Unknown')


### Category name
df1['category_name'] = df1['category_name'].fillna('Unknown/Unknown/Unknown')


## Split objects into multiple columns 
df1[['department', 'category', 'class']] = df1['category_name'].str.split('/', n=2, expand=True)


## Encode categories
cols_cats = ['department', 'category', 'class', 'brand_name']
df1[cols_cats] = df1[cols_cats].apply(lambda x: x.astype('category'))


## Description of columns
df1.columns
df1.dtypes

#find unique counts
df1.apply(lambda x: x.nunique(dropna=False))
df1[cols_cats].apply(lambda x: x.nunique(dropna=False))

#train_id: id column (int)
#name: description of product (string)
#item_condition_id: condition of product (1 = new, 2 = fairly new, 3 = good, 4 = bad, 5 = very poor)
#brand_name: brand (string; 4810 unique brands)
#price: price product was sold for (float)
#shipping: 1 = paid by seller and 0 = paid by buyer
#item_description: full description of item
#department/category/class: make up category_name (11 departments, 114 categories, 872 classes)


## Find lengths and word counts of name and item_description fields
df1['name_wc'] = df1['name'].apply(lambda x: len(x.split(' ')))
df1['desc_wc'] = df1['item_description'].apply(lambda x: len(x.split(' ')))

df1['name_len'] = df1['name'].str.len()
df1['desc_len'] = df1['item_description'].str.len()

#get min and max of each
cols_num_text = ['name_wc', 'desc_wc', 'name_len', 'desc_len']
df1[cols_num_text].apply(lambda x: str(x.min()) + '-' + str(x.max()))


## Add top 5 known categories + remainder (because of numbers of categories)
roots = ['dpt', 'cat', 'class', 'brand']

cats_roots = dict(zip(cols_cats, roots))

for col, base in cats_roots.items():
  df1 = add_top_cat(df=df1, var=col, n=5, root=base)


## Encode top 5s
cols_cats_t5 = ['dpt_top5', 'cat_top5', 'class_top5', 'brand_top5']
df1[cols_cats_t5] = df1[cols_cats_t5].apply(lambda x: x.astype('category'))


## Log-transform price
df1['price_log'] = np.log1p(df1['price'])



# Exploratory Data Analysis=========================================================================
df = df1.copy()
sns.set_theme(style="whitegrid")

## Explore counts------------------------------
### Explore derived columns from category_name
df.value_counts('department') #women most with 664,485
df.value_counts('category') #athletic apparel most with 134,383
df.value_counts('class') #pants, tights, and leggings most with 60,177


### Explore derived word count and length columns
df[cols_num_text].agg(['min', 'max'])
#name_wc: 1-17
#desc_wc: 1-245
#name_len: 1-43
#desc_len: 1-1046


## Univariate Plots------------------------------
### 1. Item condition id and shipping count data
#create lists for labels 
labs_ici = ['New', 'Fairly \nnew', 'Good', 'Bad', 'Very \npoor']
labs_ship = ['Buyer', 'Seller']

fig, axes = plt.subplots(2, 1)

make_countplot(df=df, var='item_condition_id', xlabs=labs_ici, ylim=900000, ax=axes[0])
make_countplot(df=df, var='shipping', xlabs=labs_ship, ylim=900000, ax=axes[1])

fig.suptitle('Numbers of items by Item condition id and Shipping', fontsize=11)
fig.supylabel('No. of items', fontsize=10)

plt.tight_layout()
fig.subplots_adjust(left=0.2, hspace=0.3)
plt.show()
plt.close()


### 2. Department, category, class, and brand name
#### Create orders and labels
#department
labs_dpt = ord_dpt = get_top_cat(df, 'department', 5, True)

#category
ord_cat = get_top_cat(df, 'category', 5, True)
labs_cat =  ["Athletic \nApparel", 'Makeup', "Tops & \nBlouses", 'Shoes', 'Jewelry',
             'Remaining']

#class
ord_class = get_top_cat(df, 'class', 5, True)
labs_class = ["Pants, Tights, \nLeggings", 'Face', 'T-Shirts', 'Shoes', 'Games', 
              'Remaining']

#brand_name
ord_brand = get_top_cat(df, 'brand_name', 5, True)
labs_brand = ['PINK', 'Nike', "Victoria's \nSecret", 'LuLaRoe', 'Apple', 'Remaining']


#### Make plots
fig, axes = plt.subplots(2, 2)

make_countplot(df=df, var='dpt_top5', xlabs=labs_dpt, order_bar=ord_dpt, scale='log', 
               rotate=45, ax=axes[0, 0])
make_countplot(df=df, var='cat_top5', xlabs=labs_cat, order_bar=ord_cat, scale='log',
               lableft=False, rotate=45, ax=axes[0, 1])
make_countplot(df=df, var='class_top5', xlabs=labs_class, order_bar=ord_class, scale='log',
               rotate=45, ax=axes[1, 0])
make_countplot(df=df, var='brand_top5', xlabs=labs_brand, order_bar=ord_brand, scale='log',
               lableft=False, rotate=45, ax=axes[1, 1])

fig.suptitle('Numbers of items by five most common and remaining Department, Category, Class, and Brand Name', 
             fontsize=11)
fig.supylabel('No. of items', fontsize=10)

fig.subplots_adjust(bottom=0.2, hspace=0.7, wspace=0.1)

plt.show()
plt.close()


### 3. Word counts and lengths of name and item description
fig, axes = plt.subplots(2, 2)

make_histplot(df=df, var='name_wc', xtitle='Name (word count)',
              ax=axes[0, 0])
make_histplot(df=df, var='name_len',xtitle='Name (length)',
              ax=axes[0, 1])
make_histplot(df=df, var='desc_wc', 
              xtitle='Item description (word count)', ax=axes[1, 0])
make_histplot(df=df, var='desc_len', xtitle='Item description (length)',
              ax=axes[1, 1])
               
fig.subplots_adjust(left=0.4, hspace=0.5, wspace=0.7)
fig.suptitle("Numbers of items by Name (word count & length) and Item description (word count & length)", 
             fontsize=11)
fig.supylabel("No. of items", fontsize=10)

plt.tight_layout()
plt.show()
plt.close()
#name (word count): symmetrical
#name (length): bimodal with 2nd mode near max
#desc (word count & length): right-skewed


### 4. Price (linear and log scales)
#### Figure
fig, axes = plt.subplots(2, 1)

make_histplot(df=df, var='price', xtitle='Price', ax=axes[0])
make_histplot(df=df, var='price_log', xtitle='Price (log-transformed)', ax=axes[1])

fig.suptitle("Numbers of items by Price (raw and log-transformed)",
             fontsize=11)
fig.supylabel("No. of items", fontsize=10)

plt.tight_layout()
plt.show()
plt.close()


#### Quantiles
np.quantile(df['price'], 0.25) #10
np.quantile(df['price'], 0.5) #17
np.quantile(df['price'], 0.75) #29
np.quantile(df['price'], .99) #170
np.quantile(df['price'], 1) #2009


## Bivariate Plots------------------------------
### With price
#### ICI and Shipping with price and price_log
fig, axes = plt.subplots(2, 2)

make_barplot(df=df, var='item_condition_id', y='price', xlabs=labs_ici, 
             ytitle='Mean Price ($) (\u00B1 1 SE)', alpha=0.5, ax=axes[0,0])
make_barplot(df=df, var='item_condition_id', y='price_log', xlabs=labs_ici, 
             ytitle='Mean Price ($), \nLog-Transformed (\u00B1 1 SE)', alpha=0.5, ax=axes[0, 1])
make_barplot(df=df, var='shipping', y='price', xlabs=labs_ship, 
             ytitle='Mean Price ($) (\u00B1 1 SE)', alpha=0.5, ax=axes[1,0])
make_barplot(df=df, var='shipping', y='price_log', xlabs=labs_ship, 
             ytitle='Mean Price ($), \nLog-Transformed (\u00B1 1 SE)',
             alpha=0.5, ax=axes[1,1])

fig.suptitle("Average Prices (raw and log-transformed) by Item Condition Id and Shipping category",
             fontsize=11)

plt.tight_layout()
plt.show()
plt.close()


#### Category names & brands with price
#raw prices
fig, axes = plt.subplots(2, 2)

make_barplot(df=df, var='dpt_top5', y='price', xlabs=labs_dpt, order_bar=ord_dpt, 
             xtitle='Department', alpha=0.5, rotate=45, ax=axes[0, 0])
make_barplot(df=df, var='cat_top5', y='price', xlabs=labs_cat,order_bar=ord_cat,
             xtitle='Category', alpha=0.5, rotate=45, ax=axes[0, 1])
make_barplot(df=df, var='class_top5', y='price', xlabs=labs_class, order_bar=ord_class,
             xtitle='Class', alpha=0.5, rotate=45, ax=axes[1, 0])
make_barplot(df=df, var='brand_top5', y='price', xlabs=labs_brand, order_bar=ord_brand,
             xtitle='Brand name', alpha=0.5, rotate=45, ax=axes[1, 1])

fig.suptitle('Average prices by Department, Category, Class, and Brand Name (5 most common and remaining)', 
             fontsize=11)
fig.supylabel('Mean Price ($) (\u00B1 1 SE)', fontsize=10)

fig.subplots_adjust(bottom=0.2, hspace=0.5, wspace=0.3)

plt.show()
plt.close()


#log-transformed prices
fig, axes = plt.subplots(2, 2)

make_barplot(df=df, var='dpt_top5', y='price_log', xlabs=labs_dpt, order_bar=ord_dpt, 
             xtitle='Department', alpha=0.5, rotate=45, ax=axes[0, 0])
make_barplot(df=df, var='cat_top5', y='price_log', xlabs=labs_cat,order_bar=ord_cat,
             xtitle='Category', alpha=0.5, rotate=45, ax=axes[0, 1])
make_barplot(df=df, var='class_top5', y='price_log', xlabs=labs_class, order_bar=ord_class,
             xtitle='Class', alpha=0.5, rotate=45, ax=axes[1, 0])
make_barplot(df=df, var='brand_top5', y='price_log', xlabs=labs_brand, order_bar=ord_brand,
             xtitle='Brand name', alpha=0.5, rotate=45, ax=axes[1, 1])

fig.suptitle('Average log-transformed prices by Department, Category, Class, and Brand Name \n(5 most common and remaining)', 
             fontsize=11)
fig.supylabel('Mean Log-Transformed Price ($) (± 1 SE)', fontsize=10)

fig.subplots_adjust(bottom=0.2, hspace=0.7, wspace=0.3)

plt.show()
plt.close()


#### Word counts and lengths of name and item_desc with price
#create function
def make_scatter(df, var, y, ax, ylim=None, xtitle=None, ytitle=None, lableft=True):
  sns.scatterplot(df, x=var, y=y, color=fixed_colors[var], 
                  alpha=0.05, ax=ax)
  ax.set_xlabel(xtitle, fontsize=8)
  ax.set_ylabel(ytitle, fontsize=8)
  ax.tick_params(axis='x', labelsize=8)
  ax.tick_params(axis='y', labelleft=lableft, labelsize=8)
  ax.set_ylim(0, ylim)
  

##### Raw price
fig, axes = plt.subplots(2, 2)

make_scatter(df, var='name_wc', y='price', xtitle='Name (word count)', ylim=2200, ax=axes[0,0])
make_scatter(df, var='name_len', y='price', xtitle='Name (length)', ylim=2200, lableft=False, ax=axes[0,1])
make_scatter(df, var='desc_wc', y='price', xtitle='Description (word count)', ylim=2200, ax=axes[1,0])
make_scatter(df, var='desc_len', y='price', xtitle='Description (length)', ylim=2200, lableft=False, 
             ax=axes[1,1])

fig.suptitle('''Relationships between name (word count & length) and item description 
              (word count & length) with price''', fontsize=11)
fig.supylabel('Price ($)', fontsize=10)

fig.subplots_adjust(hspace=0.5, wspace=0.5)
plt.show()
plt.close()


##### Log price
fig, axes = plt.subplots(2, 2)

make_scatter(df, var='name_wc', y='price_log', xtitle='Name (word count)', ylim=8, ax=axes[0,0])
make_scatter(df, var='name_len', y='price_log', xtitle='Name (length)', ylim=8, lableft=False, ax=axes[0,1])
make_scatter(df, var='desc_wc', y='price_log', xtitle='Description (word count)', ylim=8, ax=axes[1,0])
make_scatter(df, var='desc_len', y='price_log', xtitle='Description (length)', ylim=8, lableft=False, 
             ax=axes[1,1])

fig.suptitle('''Relationships between name (word count & length) and item description 
              (word count & length) with price''', fontsize=11)
fig.supylabel('Log-Transformed Price ($)', fontsize=10)

fig.subplots_adjust(hspace=0.5, wspace=0.5)

plt.show()
plt.close()


##### Correlations
corr_price = df[cols_num_text + ['price']].corr('spearman')['price'].drop('price')
corr_price
# name_wc     0.052304
# desc_wc     0.068076
# name_len    0.049815
# desc_len    0.068376

corr_price_log = df[cols_num_text + ['price_log']].corr('spearman')['price_log'].drop('price_log')
corr_price_log
# name_wc     0.052304 --
# desc_wc     0.068076 
# name_len    0.049815
# desc_len    0.068376  --


## Predictors only
cols_pred_num = ['name_wc', 'desc_wc', 'name_len', 'desc_len']
labs_pred_num = ['Name \n(wc)', 'Desc \n(wc)', 'Name \n(len)', 'Desc \n(len)']
df_num = df[cols_pred_num].corr()

name_desc_hmap = sns.heatmap(df_num, annot=True, cmap='coolwarm')
name_desc_hmap.set_xticklabels(labs_pred_num)
name_desc_hmap.set_yticklabels(labs_pred_num)

plt.show()
plt.close()
#unsurprisingly, desc_wc and desc_len are highly correlated
#name_len and name_wc are strongly correlated 


## Multivariate------------------------------
### ICI x Shipping on price
title_mult = 'Average price by item condition id and shipping payer (0 = buyer, 1 = shipper)'

ici_ship_price = sns.catplot(data=df, x='item_condition_id', 
                            color=fixed_colors['item_condition_id'], y='price', row='shipping',
                            kind='bar')
            
ici_ship_price.set_axis_labels("Item condition id", '')
ici_ship_price.set_xticklabels(labs_ici)
ici_ship_price.fig.supylabel('Mean Price ($) (\u00B1 1 SE)', fontsize=10)
ici_ship_price.fig.suptitle(title_mult, fontsize=11)
ici_ship_price.fig.subplots_adjust(top=0.88, bottom=0.15, hspace=0.15)

plt.show()
plt.close()
#ICI 1 has lowest avg price with shipping = 1 and 2nd highest when
  #shipping = 0
  

### ICI x Shipping on log price
title_mult2 = 'Average log-transformed price by item condition id and shipping payer (0 = buyer, 1 = shipper)'

ici_ship_lprice = sns.catplot(data=df, x='item_condition_id', 
                            color=fixed_colors['item_condition_id'], y='price_log', row='shipping',
                            kind='bar')

ici_ship_lprice.set_axis_labels("Item condition id", '')
ici_ship_lprice.set_xticklabels(labs_ici)
ici_ship_lprice.fig.supylabel('Mean Log-Transformed Price ($) (± 1 SE)', fontsize=10)
ici_ship_lprice.fig.suptitle(title_mult2, fontsize=11)
ici_ship_lprice.fig.subplots_adjust(top=0.88, bottom=0.15, hspace=0.15)

plt.show()
plt.close()
#ICI 1 has lowest avg price with shipping = 1 and 2nd highest when
  #shipping = 0


### Department x Shipping on price
title_mult3 = 'Average price by department and shipping payer (0 = buyer, 1 = shipper)'
dpt_ship_price = sns.catplot(data=df, x='dpt_top5', y='price', color=fixed_colors['dpt_top5'], row='shipping',
                             kind='bar', order=ord_dpt, alpha=0.6)

dpt_ship_price.set_axis_labels('Department', '')
dpt_ship_price.fig.supylabel('Mean Price ($) (\u00B1 1 SE)', fontsize=10)
dpt_ship_price.fig.suptitle(title_mult3, fontsize=11)
dpt_ship_price.fig.subplots_adjust(top=0.88, bottom=0.15, hspace=0.15)

plt.show()
plt.close()
#note: electronics and men flipped in order when conditioned on shipping


### Department x Shipping on log price
title_mult4 = 'Average log-transformed price by department and shipping payer (0 = buyer, 1 = shipper)'
dpt_ship_price = sns.catplot(data=df, x='dpt_top5', y='price_log', color=fixed_colors['dpt_top5'], row='shipping',
                             kind='bar', order=ord_dpt, alpha=0.6)

dpt_ship_price.set_axis_labels('Department', '')
dpt_ship_price.fig.supylabel('Mean Log-Transformed Price ($) (± 1 SE)', fontsize=10)
dpt_ship_price.fig.suptitle(title_mult3, fontsize=11)
dpt_ship_price.fig.subplots_adjust(top=0.88, bottom=0.15, hspace=0.15)

plt.show()
plt.close()



# Save DF===========================================================================================
pd.to_pickle(df, "train_clean.pkl")


