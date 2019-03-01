import warnings; warnings.simplefilter('ignore')
import os
import math
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from joblib import dump, load
from sklearn.multioutput import MultiOutputRegressor
import sklearn.metrics as metrics

params = {
  'boosting_type': 'gbdt',
  'objective': 'regression',
  'num_iterations': 400,
  'learning_rate': 0.01,
  'max_depth': 100,
  'num_leaves': 30,
  'min_data_in_leaf': 512,
  'verbose': 0,
  'n_jobs': 4,
}

filename = Path('./GBM_model.joblib')

## Train Model
# Trains multiple regressors for direct multistep forecast strategy
def train(config, data):
  if (os.path.exists(filename)):
    return
  else:
    gbm = MultiOutputRegressor(lgb.LGBMRegressor(**params))
    model = gbm.fit(data['X_train'], data['y_train'])
    dump(model, filename)

## Predict
def predict(config, data):
  model = load(filename)
  pred_np = model.predict(data['X_test'])
  pred_df = pd.DataFrame(index=data['X_test'].index)
  pred_df['GBM'] = pred_np[::config['window'], :].flatten()
  pred_df['True'] = data['y_test'].iloc[::config['window'], :].values.flatten()
  rmse = math.sqrt(metrics.mean_squared_error(pred_df['True'], pred_df['GBM']))
  r2 = metrics.r2_score(pred_df['True'], pred_df['GBM'])
  return pred_df, rmse, r2

## Getter Feature Importance
def get_feature_importance(config, data):
  # Returns array of importances due to multi output regressor
  model = load(filename)
  feature_importance = [pd.Series(e.feature_importances_, index=data['X_train'].columns).sort_values() for e in model.estimators_]
  return pd.concat(feature_importance, axis=1)
