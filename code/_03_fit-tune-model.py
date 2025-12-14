# This script creates the encoding pipelines and conducts cross-validation with model tuning

# Load Libraries, Data, and Functions===============================================================
## Load libraries
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer, OneHotEncoder
from sklearn.model_selection import KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error, make_scorer
from sklearn.model_selection import RandomizedSearchCV
import joblib


## Data
ROOT = Path.cwd()
data_path_in = ROOT / "data" / "train_clean_feat.pkl"
df = pd.read_pickle(data_path_in)


## Functions
from code._00_helper_objs_fns import (
  make_ridge_pipeline,
  make_tree_pipeline,
  make_xgb_pipeline
)



# Feature Engineering Pipeline======================================================================
#-----see _00 script for below steps as they are functionalized-----

## Steps: 
  # 1) Create sets of pass-through columns
  # 2) Custom Transformers
  # 3) Pipeline Pieces
  # 4) Pipelines for Each Model Family


## Step 5: Cross-Validation--------------------
### Extract X and y
X = df.drop(columns=['price_log', 'price'])
y = df['price_log']


### Create 3 folds (large dataset, low RAM)
cv = KFold(n_splits=3, shuffle=True, random_state=42) 

#score using RMSE
rmsle_equiv_scorer = make_scorer(root_mean_squared_error, greater_is_better=False)


### Ridge regression----- 
#### Run 3 ridge models at different alphas
#alpha = 0.1
ridge_model1 = Ridge(alpha=0.1, solver='auto')
ridge_pipe1=make_ridge_pipeline(ridge_model1)
ridge_scores1=cross_val_score(ridge_pipe1, X, y, cv=cv, scoring=rmsle_equiv_scorer)
ridge_scores1 
ridge_scores1.mean() 
#newest 0.50019 (25K/30K)
#newer: 0.50546 #new + increased TF-IDF name and desc features by 2K each
#new: 0.5096 #new boolean keyword matching field in item description
#old: 0.5098

#alpha = 1.0
ridge_model2 = Ridge(alpha=1.0, solver='auto')
ridge_pipe2=make_ridge_pipeline(ridge_model2)
ridge_scores2=cross_val_score(ridge_pipe2, X, y, cv=cv, scoring=rmsle_equiv_scorer)
ridge_scores2 
ridge_scores2.mean() 
#0.4960 (20K/50K)
#0.49589 (25K/30K) <--
#0.49663 (20K/25K)
#0.49738 (17K/22K)
#0.49814 (15K/20K)
#0.49919 (13K/18K)
#0.5005 (11K/16K)
#0.5014 (10K/15K)
#0.5024 (9K/14K)
#0.5036 (8K/13K)
#newer: 0.50498 (7K/12K) #best
#new: 0.5093 (5K/10K + new boolean fields)
#old: 0.5095 (5K/10K)

#alpha = 10.0
ridge_model3 = Ridge(alpha=10.0, solver='auto')
ridge_pipe3=make_ridge_pipeline(ridge_model3)
ridge_scores3=cross_val_score(ridge_pipe3, X, y, cv=cv, scoring=rmsle_equiv_scorer)
ridge_scores3 
ridge_scores3.mean() 
#newest: 0.49793 (25K/30K)
#newer: 0.50524
#new: 0.5095
#old: 0.5096

#newer: Best: alpha = 1.0 -> RMSLE = 0.505498
#new: Best: alpha = 1.0 -> RMSLE = 0.5093
#old: Best: alpha = 1.0 -> RMSLE = 0.5095


### Tree model example (set for older machine)
# tree_model = RandomForestRegressor(random_state=42)
tree_model = RandomForestRegressor(
    n_estimators=50,       # Reduced number of trees
    max_depth=10,          # Limited tree depth
    min_samples_leaf=5,    # Increased minimum samples per leaf
    max_features='sqrt',   # Limited features for splitting
    n_jobs=-1,             # Use all available CPU cores
    random_state=42        # For reproducibility
)
tree_pipe = make_tree_pipeline(tree_model)

#score using RMSE
tree_scores = cross_val_score(tree_pipe, X, y, cv=cv, scoring=rmsle_equiv_scorer)
tree_scores
tree_scores.mean()
#new: 0.6342
#old: 0.6366


### XGBoost example (set for older machine)
xgb_model = XGBRegressor(
    tree_method='hist',
    max_depth=4,
    n_estimators=100,
    n_jobs=4,  
    learning_rate=0.2,
    subsample=0.7,
    random_state=42 # For reproducibility
)
xgb_pipe = make_xgb_pipeline(xgb_model)

xgb_scores = cross_val_score(xgb_pipe, X, y, cv=cv, scoring=rmsle_equiv_scorer)
xgb_scores
xgb_scores.mean()
#newer: 0.5858 #new + +3 K/+4 K and +50/+50
#new: 0.5918 #new boolean keyword fields
#old: 0.5925


# Hyperparameter Tuning=============================================================================
## Given the RMSLE scores using specified settings with 3 folds...

  #best
  
  
  
  #second and third runs...
  #RR: 0.50546
  #RF: 0.6342
  #XGB: 0.5858
#...tune RR (try new alphas) & XGB models
  
  #original run...
  #Ridge Regression: 0.5095
  #RF: 0.637
  #XGB: 0.5925
#...tune RR & XGB models


## Ridge Regression tuning
#try more alphas
#alpha = 0.01
ridge_model4 = Ridge(alpha=0.01, solver='auto')
ridge_pipe4 = make_ridge_pipeline(ridge_model4)
ridge_scores4=cross_val_score(ridge_pipe4, X, y, cv=cv, scoring=rmsle_equiv_scorer)
ridge_scores4 
ridge_scores4.mean() 
#newest: 0.50172 (25K/30K)
#newer: 0.50554
#old: 0.5098


#alpha = 100.0
ridge_model5 = Ridge(alpha=100.0, solver='auto')
ridge_pipe5=make_ridge_pipeline(ridge_model5)
ridge_scores5=cross_val_score(ridge_pipe5, X, y, cv=cv, scoring=rmsle_equiv_scorer)
ridge_scores5 
ridge_scores5.mean() 
#newest: 0.5229 (25K/30K)
#Newer: 0.5229 (best is still alpha = 0.1 with RMSLE = 0.50546)
#Old: 0.5243 (best is still alpha = 1.0 with RMSLE = 0.5095)


#choose best ridge regression model
ridge_dict = {'ridge_1': ridge_scores1.mean(),
              'ridge_2': ridge_scores2.mean(),
              'ridge_3': ridge_scores3.mean(),
              'ridge_4': ridge_scores4.mean(),
              'ridge_5': ridge_scores5.mean()
              }

ridge_best_model = max(ridge_dict, key=ridge_dict.get)
ridge_best_model 

best_ridge = ridge_pipe2


## XGB tuning
### Define pipeline
xgb_model = XGBRegressor(
  tree_method='hist', 
  n_jobs=4,
  random_state=42
)
xgb_pipe=make_xgb_pipeline(xgb_model)


### Define hyperparameter search space
param_dist = {
  'model__max_depth': [3, 4, 5],
  'model__learning_rate': [0.05, 0.1],
  'model__n_estimators': [300, 600],
  'model__subsample': [0.7, 1.0],
  'model__colsample_bytree': [0.6, 0.8],
  'model__reg_alpha': [0, 0.1],
  'model__reg_lambda': [1, 2] 
}


### RandomizedSearchCV
random_search = RandomizedSearchCV(
  estimator=xgb_pipe,
  param_distributions=param_dist,
  n_iter=8, 
  scoring=rmsle_equiv_scorer,
  cv=3, 
  verbose=2,
  random_state=42,
  n_jobs=1
)


### Run search
random_search.fit(X, y)


### Retrieve parameters & score
best_xgb=random_search.best_estimator_
-random_search.best_score_ 
random_search.best_params_
#{'model__subsample': 0.7, 'model__reg_lambda': 1, 'model__reg_alpha': 0.1, 
  #'model__n_estimators': 600, 'model__max_depth': 4, 'model__learning_rate': 0.1, 
  #'model__colsample_bytree': 0.6}
#New: 0.5625
#Old: 0.571


### Evaluate model on cross-validation or a holdout set
xgb_cv_scores=cross_val_score(best_xgb, X, y, cv=3, scoring=rmsle_equiv_scorer)
-xgb_cv_scores.mean()
#New: 0.5632
#Old: 0.571

#compare with best ridge regression (0.50546) --> best = ridge regression


## Retrain the best model on all training data
final_model = best_ridge
final_model.fit(X, y)
#this is a production-ready model trained on all available data



# Save Best Model===================================================================================
data_path_out = ROOT / "data" / 'best_model.pkl'
# joblib.dump(final_model, data_path_out)


