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
#RMSLE: 0.50019 


#alpha = 1.0
ridge_model2 = Ridge(alpha=1.0, solver='auto')
ridge_pipe2=make_ridge_pipeline(ridge_model2)
ridge_scores2=cross_val_score(ridge_pipe2, X, y, cv=cv, scoring=rmsle_equiv_scorer)
ridge_scores2 
ridge_scores2.mean() 
#0.49589 


#alpha = 10.0
ridge_model3 = Ridge(alpha=10.0, solver='auto')
ridge_pipe3=make_ridge_pipeline(ridge_model3)
ridge_scores3=cross_val_score(ridge_pipe3, X, y, cv=cv, scoring=rmsle_equiv_scorer)
ridge_scores3 
ridge_scores3.mean() 
#0.49793 

# Best RR: alpha = 1.0; RMSLE = 0.49589


### Random Forest 
#create model and pipeline
tree_model = RandomForestRegressor(
    n_estimators=50,       
    max_depth=10,         
    min_samples_leaf=5,    
    max_features='sqrt',   
    n_jobs=-1,            
    random_state=42       
)
tree_pipe = make_tree_pipeline(tree_model)

#run cross-validation
tree_scores = cross_val_score(tree_pipe, X, y, cv=cv, scoring=rmsle_equiv_scorer)
tree_scores
tree_scores.mean()
#0.6342


### XGBoost  
#create model and pipeline
xgb_model = XGBRegressor(
    tree_method='hist',
    max_depth=4,
    n_estimators=100,
    n_jobs=4,  
    learning_rate=0.2,
    subsample=0.7,
    random_state=42 
)
xgb_pipe = make_xgb_pipeline(xgb_model)

#run cross-validation
xgb_scores = cross_val_score(xgb_pipe, X, y, cv=cv, scoring=rmsle_equiv_scorer)
xgb_scores
xgb_scores.mean()
#0.5858 



# Hyperparameter Tuning=============================================================================
## Given the RMSLE scores using specified settings with 3 folds...

  #best
  #RR: 0.49589
  #RF: 0.6342
  #XGB: 0.5858
  #...tune RR & XGB models


## Ridge Regression tuning (test more alphas)
#alpha = 0.5
ridge_model4 = Ridge(alpha=0.5, solver='auto')
ridge_pipe4 = make_ridge_pipeline(ridge_model4)
ridge_scores4=cross_val_score(ridge_pipe4, X, y, cv=cv, scoring=rmsle_equiv_scorer)
ridge_scores4 
ridge_scores4.mean() 
#0.49730 


#alpha = 5.0
ridge_model5 = Ridge(alpha=5.0, solver='auto')
ridge_pipe5 = make_ridge_pipeline(ridge_model5)
ridge_scores5=cross_val_score(ridge_pipe5, X, y, cv=cv, scoring=rmsle_equiv_scorer)
ridge_scores5 
ridge_scores5.mean() 
#0.49556 [best model]


#choose best ridge regression model
ridge_dict = {'ridge_1': ridge_scores1.mean(),
              'ridge_2': ridge_scores2.mean(),
              'ridge_3': ridge_scores3.mean(),
              'ridge_4': ridge_scores4.mean(),
              'ridge_5': ridge_scores5.mean()
              }

best_ridge = max(ridge_dict, key=ridge_dict.get)
best_ridge = ridge_pipe5 #to avoid re-running all models


## XGB tuning
### Define model & pipeline
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
#0.5625


### Evaluate model on cross-validation set
xgb_cv_scores=cross_val_score(best_xgb, X, y, cv=3, scoring=rmsle_equiv_scorer)
-xgb_cv_scores.mean()
#0.5632

#compare with best ridge regression (0.49589) --> best = ridge regression


## Retrain the best model on all training data
final_model = best_ridge
final_model.fit(X, y)
#this is a production-ready model trained on all available data



# Save Best Model===================================================================================
data_path_out = ROOT / "data" / 'best_model.pkl'
# joblib.dump(final_model, data_path_out)


