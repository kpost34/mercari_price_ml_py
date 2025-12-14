# This script analyzes feature importances of selected model & evaluates it on test data

# Load Libraries, Data, and Functions===============================================================
## Load libraries
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_log_error
import pandas as pd
import numpy as np
import seaborn as sns


## Data
ROOT = Path.cwd()
data_path_in = ROOT / "data" / "train_clean_feat.pkl"
model_path_in = ROOT / "data" / 'best_model.pkl'
df = pd.read_pickle(data_path_in)
best_ridge = joblib.load(model_path_in)


## Functions
from code._00_helper_objs_fns import (
  cols_other,
  numeric_minmax,
  numeric_log,
  cat_cols,
  prepare_features
)



# Model Analysis====================================================================================
## Extract X and y fro training set--------------------
X_train_pre = df.drop(columns=['price_log', 'price'])
y_train_obs = df['price_log']


## Predicted vs actual plot--------------------
#get predictions
y_train_pred = best_ridge.predict(X_train_pre)

df_y_train = pd.DataFrame({'y_obs': y_train_obs, 
                           'y_pred': y_train_pred})

#make plot
p1 = sns.scatterplot(data=df_y_train, 
                     x='y_obs', y='y_pred',
                     alpha=0.05)
p1.plot(p1.get_xlim(), p1.get_xlim(), '--', color='black')
p1.set_xlabel('Observed Log-Transformed Price ($)')
p1.set_ylabel('Predicted Log-Transformed Price ($)')
p1.set_title('Predicted vs Observed Log Prices')
plt.show()
plt.close()


## Analyze feature importances--------------------
### Extract ridge model from pipeline
ridge = best_ridge.named_steps['model']
preprocessor = best_ridge.named_steps['preprocessor']


### Get overall text features
#fitted text transformer
text_pipe = preprocessor.named_transformers_['text']

#extract each TF-IDF vectorizer
name_tfidf = text_pipe.named_transformers_['name_tfidf']
desc_tfidf = text_pipe.named_transformers_['desc_tfidf']

#get feature names
name_features = name_tfidf.get_feature_names_out()
desc_features = desc_tfidf.get_feature_names_out()

#tag them by adding in origin field
name_features_tagged = [f"tfidf_name_{f}" for f in name_features]
desc_features_tagged = [f"tfidf_desc_{f}" for f in desc_features]


### Build feature list
#numeric features
num_features = numeric_minmax + numeric_log

#categorical features
cat_features = cat_cols

#text features with tag
text_features_tagged = name_features_tagged + desc_features_tagged

#passthrough features
pass_features = cols_other

#all features, in exact transform order
all_features = (
  num_features +
  cat_features +
  text_features_tagged +
  pass_features
)


### Get coefficients
coefs = ridge.coef_
df_coef = pd.DataFrame({
  'feature': all_features,
  'coef': coefs
})

df_coef['coef_abs'] = df_coef['coef'].abs()
cond_feat = [(df_coef['feature'].str.startswith('tfidf_desc_')),
              (df_coef['feature'].str.startswith('tfidf_name_'))]
choice_feat = ['desc', 'name']
df_coef['name_desc_other'] = np.select(cond_feat, choice_feat, default='other')

name_desc_colors = {'name': 'darkred',
                    'desc': 'darkblue'}


### Plot results
#### TF-IDF features
df_coef_name_desc = df_coef[df_coef['name_desc_other']!='other']
df_coef_name_desc['feature'] = df_coef_name_desc['feature'].str.replace('tfidf_(name|desc)_', '', regex=True)
df_coef_name_desc

df_coef_pos = df_coef_name_desc.sort_values('coef', ascending=False).iloc[0:10]
df_coef_neg = df_coef_name_desc.sort_values('coef', ascending=True).iloc[0:10]


#### Make plot
fig, axes = plt.subplots(1, 2)

sns.barplot(data=df_coef_neg, y='feature', x='coef', hue='name_desc_other',
            palette=name_desc_colors, ax=axes[0], legend=False)
sns.barplot(data=df_coef_pos, y='feature', x='coef', hue='name_desc_other',
            palette=name_desc_colors, ax=axes[1])

axes[0].set_xlabel('')
axes[0].set_ylabel('')

axes[1].set_xlabel('')
axes[1].set_ylabel('')

fig.suptitle("Top 10 positive and negative coefficients of TF-IDF features of final model on training data",
             fontsize=11)
fig.supxlabel("Coefficient")
fig.supylabel("Feature")

axes[1].legend(
    loc="center left",
    bbox_to_anchor=(1, 0.5),
    frameon=False
)

# plt.tight_layout()
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.show()
plt.close()


#### Non-TF-IDF features
#filter and sort DF
df_coef_other = df_coef[df_coef['name_desc_other']=='other'].sort_values('coef', ascending=False)


#make plot
p3 = sns.barplot(df_coef_other, y='feature', x='coef', color='darkgreen', legend=False)
p3.axvline(x=0, color="black", linestyle="-", linewidth=1)

p3.set_xlabel('Coefficient')
p3.set_ylabel('Feature')
p3.set_title('Coefficients of non-TF-IDF features of final model \non training data')

plt.tight_layout()
plt.show()
plt.close()


# Run Model Predictions=============================================================================
## Import test data
data_path_in2 = ROOT / "data" / "test.tsv"
df_test0 = pd.read_csv(data_path_in2, sep='\t')


## Prepare test data
X_test_pre = prepare_features(df_test0)


### Predict on preprocessed data
y_pred = best_ridge.predict(X_test_pre)


### Check preds
df_pred = pd.DataFrame({'price_log_pred': y_pred})
df_test_pred = pd.concat([X_test_pre, df_pred], axis=1)
df_test_pred.head()
