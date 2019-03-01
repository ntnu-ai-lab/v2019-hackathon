import warnings; warnings.simplefilter('ignore')
import os
import math
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import dump, load
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import sklearn.metrics as metrics

params = {
  'n_estimators': 100,
  'max_features': 'sqrt',
  'max_depth': 100,
  'min_samples_split': 5,
  'min_samples_leaf': 4,
  'bootstrap': True,
  'verbose': 0,
}

filename = Path('./RF_model.joblib')

## Train if no saved model
# Trains multiple regressors for direct multistep forecast strategy
def train(config, data):
  if (os.path.exists(filename)):
    return
  else:
    rf = MultiOutputRegressor(RandomForestRegressor(**params))
    model = rf.fit(data['X_train'], data['y_train'])
    dump(model, filename)

## Predict
def predict(config, data):
  model = load(filename)
  pred_np = model.predict(data['X_test'])
  pred_df = pd.DataFrame(index=data['X_test'].index)
  pred_df['RF'] = pred_np[::config['window'], :].flatten()
  pred_df['True'] = data['y_test'].iloc[::config['window'], :].values.flatten()
  rmse = math.sqrt(metrics.mean_squared_error(pred_df['True'], pred_df['RF']))
  r2 = metrics.r2_score(pred_df['True'], pred_df['RF'])
  return pred_df, rmse, r2

## Getter Feature Importance
def get_feature_importance(config, data):
  model = load(filename)
  feature_importance = [pd.Series(e.feature_importances_, index=data['X_train'].columns).sort_values() for e in model.estimators_]
  return pd.concat(feature_importance, axis=1)
