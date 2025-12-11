# Script contains functions to help with coding

# Load Packages and Create Color Dictionary=========================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


fixed_colors = {
  'item_condition_id':'darkblue',
  'shipping': 'darkorange',
  'department': 'darkred',
  'dpt_top5': 'darkred',
  'category': 'purple',
  'cat_top5': 'purple',
  'class': 'brown',
  'class_top5': 'brown',
  'brand_name': 'goldenrod',
  'brand_top5': 'goldenrod',
  'name_wc': 'steelblue',
  'name_len': 'teal',
  'desc_wc': 'tomato',
  'desc_len': 'coral',
  'price': 'darkgreen',
  'price_log': 'green',
  'has_cat_name': 'blue',
  'has_brand': 'blue',
  'has_desc': 'blue',
  'has_keyword_new': 'blue'
}


# Feature Engineering Functions=====================================================================
## Functions to get top 5 known categories and add them + remaining to DF
def get_top_cat(df, var, n, remain=False):
  df_n = df.value_counts(var).reset_index()
  t_top_cat = df_n[~df_n[var].isin(['Unknown', 'Other'])].iloc[:n][var].tolist()
  if remain:
    t_top_cat = t_top_cat + ['Remaining']
  return t_top_cat

def add_top_cat(df, var, n, root):
  var_top_n = root + '_top' + str(n)
  t_top_cat = get_top_cat(df, var, n)
  df[var_top_n] = np.where(df[var].isin(t_top_cat),
                           df[var], 'Remaining')
  return df


## Function to map average price to categories for a new feature
def map_avg_price(df, var):
  new_var = var + '_lprice'
  mapping = df.groupby(var)['price_log'].mean().to_dict()
  df[new_var] = df[var].map(mapping).astype(float)
  return df


## Function to conduct a log transformation
def log_transform(x):
    return np.log1p(x)


## Function to compute correlation ratio
def correlation_ratio(categories, values):
    """
    Compute the correlation ratio (eta) between a categorical and a numerical variable.
    
    Parameters
    ----------
    categories : array-like
        Categorical variable (e.g. strings, ints, or categories).
    values : array-like
        Numeric variable.
    
    Returns
    -------
    eta : float
        Correlation ratio between 0 and 1.
    """
    # Convert to arrays
    categories = np.array(categories)
    values = np.array(values)

    # Compute group means and overall mean
    category_means = []
    category_counts = []
    for cat in np.unique(categories):
        cat_values = values[categories == cat]
        category_means.append(np.mean(cat_values))
        category_counts.append(len(cat_values))
    
    overall_mean = np.mean(values)

    # Between-group and total variance
    numerator = np.sum(category_counts * (category_means - overall_mean) ** 2)
    denominator = np.sum((values - overall_mean) ** 2)

    # Return eta (sqrt of ratio)
    return np.sqrt(numerator / denominator) if denominator != 0 else 0.0



# EDA Functions=====================================================================================
## Function to make countplots
def make_countplot(ax, df, var, xlabs, order_bar=None, scale=None, ylim=None, lableft=True, 
                   rotate=0):
  # Set xtitle
  xtitle = var.replace('_', ' ').capitalize()
  
  # Create plot and labels
  sns.countplot(data=df, x=var, ax=ax, 
                color=fixed_colors[var], order=order_bar)
  ax.set_xlabel(xtitle, fontsize=10)
  ax.set_ylabel('')
  ax.set_xticklabels(xlabs)
  ax.tick_params(axis='x', rotation=rotate, labelsize=8)
  ax.tick_params(axis='y', labelleft=lableft, labelsize=8)
  
  #Conditionally add scale and ylimit
  if scale is not None:
    ax.set_yscale(scale)
    ax.set_ylim(1, ylim)
  else:
    ax.set_ylim(0, ylim)


## Function to make histograms
def make_histplot(ax, df, var, xtitle=None):
  # Create plot and labels
  sns.histplot(data=df, x=var, color=fixed_colors[var], ax=ax)
  ax.set_ylabel('')
  ax.tick_params(axis='x', labelsize=8)
  ax.tick_params(axis='y', labelsize=8)
  
  # Conditionally add xlabel
  if xtitle is None:
    ax.set_xlabel(var, fontsize=10)
  else:
    ax.set_xlabel(xtitle, fontsize=10)
  

## Function to make barplot
def make_barplot(ax, df, var, y, xlabs, order_bar=None, xtitle=None, ytitle=None, alpha=1, bar='se', 
                 scale=None, ylim=None, lableft=True, rotate=0):
  # Conditionally create xtitle
  if xtitle is None:
    xlab = var.replace('_', ' ').capitalize()
  else:
    xlab = xtitle

  # Create plot and add labels
  sns.barplot(data=df,
              x=var, y=y,
              order=order_bar, color=fixed_colors[var], alpha=alpha,
              errorbar=bar,
              ax=ax)
  ax.set_xlabel(xlab, fontsize=8)
  ax.set_ylabel(ytitle, fontsize=8)
  ax.set_xticklabels(xlabs)
  ax.tick_params(axis='x', rotation=rotate, labelsize=8)
  ax.tick_params(axis='y', labelleft=lableft, labelsize=8)
  
  # Conditionally add scale and ylimit
  if scale is not None:
    ax.set_yscale(scale)
    ax.set_ylim(1, ylim)
  else:
    ax.set_ylim(0, ylim)


## Function to make scatter plot
def make_scatter(df, var, y, ax, ylim=None, xtitle=None, ytitle=None, lableft=True):
  sns.scatterplot(df, x=var, y=y, color=fixed_colors[var], 
                  alpha=0.05, ax=ax)
  ax.set_xlabel(xtitle, fontsize=8)
  ax.set_ylabel(ytitle, fontsize=8)
  ax.tick_params(axis='x', labelsize=8)
  ax.tick_params(axis='y', labelleft=lableft, labelsize=8)
  ax.set_ylim(0, ylim)



# Transformer & Encoder=============================================================================
## Top-5 Binning Transformer
class TopKCategories(BaseEstimator, TransformerMixin):
  
  #initialize object and set hyperparameters
  def __init__(self, top_k=5, exclude_labels=('Unknown', 'Other')):
      self.top_k = top_k ## of top categories to retain by frequency
      self.exclude_labels = exclude_labels  #values to ignore when counting frequency
      self.top_categories_ = {} #later holds learned top-K categories per col
      
  #transformer learns which categories are most frequent
  def fit(self, X, y=None):
      X = pd.DataFrame(X) #ensures X is always a DF
      exclude = set(self.exclude_labels)  #convert tuple into a set to speed up membership checks
      self.top_categories_ = {}
      for col in X.columns:
          counts = X[col].value_counts() #get freq of each category
          valid = counts[~counts.index.isin(exclude)] #drop excluded labels
          top = valid.nlargest(self.top_k).index.tolist() #select most frequent top_k categories
          self.top_categories_[col] = top #stores top_k categories
      return self #returns self so it's compatible with Pipeline
  #object 'knows' which categories are 'top K' for each column
  
  #apply learned top-K mapping to new data
  def transform(self, X):
      X = pd.DataFrame(X).copy() #convert input to DF and copy it (to avoid changing original data)
      for col in X.columns:
          X[col] = np.where( 
              #if category is a top-K category, keep it as is, otherwise replace with 'Remaining'
              X[col].isin(self.top_categories_[col]), X[col], 'Remaining'
          )
      return X


## Target Mean Encoder (per fold)
class TargetMeanEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y):
        #ensures that the inputs are pandas objects
        X = pd.DataFrame(X)
        y = pd.Series(y)
        #for each categorical column in X...
        self.encodings_ = {
            #group y by each category in col, compute mean y, convert to dict
            col: y.groupby(X[col]).mean().to_dict() for col in X.columns
        }
        self.global_mean_ = y.mean()
        return self

    #convert each category to its target mean value
    def transform(self, X):
        X = pd.DataFrame(X).copy() #make a copy to avoid changing original data
        for col in X.columns:
            #replace each cat with encoded value 
            X[col] = X[col].map(self.encodings_[col])
        return X



# Encoding/feature-engineering Pipeline Preprocessors & Functions===================================
## Pass-through columns
cols_other = ['item_condition_id', 'shipping', 'has_cat_name', 'has_brand', 'has_desc', 
              'has_keyword_new']

## Preprocessors
### Numeric processing
#numeric columns
numeric_minmax = ['name_wc']
numeric_log = ['desc_len']

#categorical columns
cat_cols = ['brand_name', 'department', 'category', 'class']

numeric_preproc = ColumnTransformer([ 
    ('minmax', MinMaxScaler(), numeric_minmax), #apply MM scaler to numeric_minmax cols
    ('log', FunctionTransformer(np.log1p), numeric_log) #apply np.log1p to all numeric_log cols
])


### Text processing
text_cols = ['name', 'item_description']

#Ridge
text_preproc_ridge = ColumnTransformer([
  ('name_tfidf', TfidfVectorizer(max_features=5000, stop_words='english'), 'name'),
  ('desc_tfidf', TfidfVectorizer(max_features=10000, stop_words='english'), 'item_description')
])


#RF
text_preproc_rf = ColumnTransformer([
    ('name_tfidf', Pipeline([('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')), 
                             ('svd', TruncatedSVD(n_components=20))]), 'name'),
    ('desc_tfidf', Pipeline([('tfidf', TfidfVectorizer(max_features=2000, stop_words='english')), 
                             ('svd', TruncatedSVD(n_components=40))]), 'item_description')
])


#XGB
text_preproc_xgb = ColumnTransformer([
    ('name_tfidf', Pipeline([('tfidf', TfidfVectorizer(max_features=2000, stop_words='english')), 
                             ('svd', TruncatedSVD(n_components=50))]), 'name'),
    ('desc_tfidf', Pipeline([('tfidf', TfidfVectorizer(max_features=4000, stop_words='english')), 
                             ('svd', TruncatedSVD(n_components=100))]), 'item_description')
])



## Function to create a liner model pipeline
def make_ridge_pipeline(model):
  #converts cat cols into numeric versions (top 5 + Remaining) & computes target mean
  cat_target_mean_pipe = Pipeline([
      ('top5', TopKCategories(top_k=5)),
      ('target_mean', TargetMeanEncoder())
  ])
  
  #combines numeric and categorical preprocessing
  preprocessor = ColumnTransformer([
      ('num', numeric_preproc, numeric_minmax + numeric_log),
      ('cat', cat_target_mean_pipe, cat_cols),
      ('text', text_preproc_ridge, text_cols),
      ('pass', 'passthrough', cols_other)
  ])
  
  #assembles the full pipeline
  return Pipeline([
      ('preprocessor', preprocessor),
      ('model', model)
  ])
#NOTE: top K categories drop and only target mean ones retained because of collinearity discussed above


## Function to create a tree-based pipeline
def make_tree_pipeline(model):
  #keeps 5 most common categories + 'Remaining' --> one hot encoding
  cat_top5_pipe = Pipeline([
      ('top5', TopKCategories(top_k=5)),
      ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
  ])
  
  #keeps 5 most common categories + 'Remaining' --> target encoding
  cat_target_mean_pipe = Pipeline([
      ('top5', TopKCategories(top_k=5)),
      ('target_mean', TargetMeanEncoder())
  ])
  
  #TargetMean only for item_condition_id
  item_cond_pipe = Pipeline([
    ('target_mean', TargetMeanEncoder())
  ])
  
  #concatenate outputs from one-hot encoding and target mean encoding branches
  cat_preproc = ColumnTransformer([
      ('top5_onehot', cat_top5_pipe, cat_cols),
      ('top5_lprice', cat_target_mean_pipe, cat_cols),
      ('item_cond_lprice', item_cond_pipe, ['item_condition_id'])
  ])
  
  #combines numerical and categorical pre-processing
  preprocessor = ColumnTransformer([
      ('num', numeric_preproc, numeric_minmax + numeric_log),
      ('cat', cat_preproc, cat_cols + ['item_condition_id']),
      ('text', text_preproc_rf, text_cols),
      ('pass', 'passthrough', cols_other)
  ])
  
  #wraps preprocessor and model into one end-to-end pipeline
  return Pipeline([
      ('preprocessor', preprocessor),
      ('model', model)
  ])


## Function to create an xgb-based pipeline
def make_xgb_pipeline(model):
  #keeps 5 most common categories + 'Remaining' --> one hot encoding
  cat_top5_pipe = Pipeline([
      ('top5', TopKCategories(top_k=5)),
      ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
  ])
  
  #keeps 5 most common categories + 'Remaining' --> target encoding
  cat_target_mean_pipe = Pipeline([
      ('top5', TopKCategories(top_k=5)),
      ('target_mean', TargetMeanEncoder())
  ])
  
  #TargetMean only for item_condition_id
  item_cond_pipe = Pipeline([
    ('target_mean', TargetMeanEncoder())
  ])
  
  #concatenate outputs from one-hot encoding and target mean encoding branches
  cat_preproc = ColumnTransformer([
      ('top5_onehot', cat_top5_pipe, cat_cols),
      ('top5_lprice', cat_target_mean_pipe, cat_cols),
      ('item_cond_lprice', item_cond_pipe, ['item_condition_id'])
  ])
  
  #combines numerical and categorical pre-processing
  preprocessor = ColumnTransformer([
      ('num', numeric_preproc, numeric_minmax + numeric_log),
      ('cat', cat_preproc, cat_cols + ['item_condition_id']),
      ('text', text_preproc_xgb, text_cols),
      ('pass', 'passthrough', cols_other)
  ])
  
  #wraps preprocessor and model into one end-to-end pipeline
  return Pipeline([
      ('preprocessor', preprocessor),
      ('model', model)
  ])


