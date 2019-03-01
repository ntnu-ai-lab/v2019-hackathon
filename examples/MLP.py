import warnings; warnings.simplefilter('ignore')
import os
import copy
import time
import math
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sklearn.metrics as metrics
from sklearn.preprocessing import MinMaxScaler

class Model(torch.nn.Module):
  def __init__(self, n_feature, n_hidden, n_output, n_hidden_layers, dropout, **_):
    super(Model, self).__init__()
    self.n_hidden_layers = n_hidden_layers
    self.input_linear = nn.Linear(n_feature, n_hidden)
    self.middle_linear = nn.Linear(n_hidden, n_hidden)
    self.output_linear = nn.Linear(n_hidden, n_output)
    self.dropout = nn.Dropout(dropout)

    #self.apply(self.init_weights)

  def init_weights(self, model):
    if type(model) == nn.Linear:
      nn.init.uniform_(model.weight, 0, 0.001)

  def apply_dropout(self):
    def apply_drops(m):
      if type(m) == nn.Dropout:
        m.train()
    self.apply(apply_drops)

  def forward(self, x):
    x = self.input_linear(x)
    for _ in range(0, self.n_hidden_layers):
      x = self.dropout(F.relu(self.middle_linear(x)))
    x = self.dropout(x)
    x = self.output_linear(x)
    return x

## Class to computes and stores the average and current value
class AverageMeter(object):
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val*n
    self.count += n
    self.avg = self.sum/self.count

## Dataset
class Dataset(object):
  def __init__(self, X, y):
    assert len(X) == len(y)
    self.X = X
    self.y = y

  def __len__(self):
    return len(self.X)

  def __getitem__(self, index):
    x = torch.Tensor(self.X[index])
    y = torch.Tensor(self.y[index])
    return x, y

## Getter for dataloaders for all datasets
def get_datasets(data, batch_size, shuffle, num_workers=0, y_scaler=None, X_scaler=None, **_):
  # Prepare for predictions with single loader
  if X_scaler and y_scaler:
    X_test = X_scaler.transform(data['X_test'])
    y_test = y_scaler.transform(data['y_test'])
    pred_generator = DataLoader(Dataset(X_test, y_test), shuffle=False, batch_size=batch_size, num_workers=num_workers)
    return None, None, None, pred_generator, None, y_scaler

  # Prepare for training with all loaders
  X_scaler = MinMaxScaler(feature_range=(0, 10))
  y_scaler = MinMaxScaler(feature_range=(0, 10))
  
  X_train = X_scaler.fit_transform(data['X_train'])
  y_train = y_scaler.fit_transform(data['y_train'])
  
  X_val = X_scaler.transform(data['X_val'])
  y_val = y_scaler.transform(data['y_val'])
  
  X_test = X_scaler.transform(data['X_test'])
  y_test = y_scaler.transform(data['y_test'])

  training_generator = DataLoader(Dataset(X_train, y_train), shuffle=shuffle, batch_size=batch_size, num_workers=num_workers)
  validation_generator = DataLoader(Dataset(X_val, y_val), shuffle=shuffle, batch_size=batch_size, num_workers=num_workers)
  test_generator = DataLoader(Dataset(X_test, y_test), shuffle=shuffle, batch_size=batch_size, num_workers=num_workers)
  pred_generator = DataLoader(Dataset(X_test, y_test), shuffle=False, batch_size=batch_size, num_workers=num_workers)

  return training_generator, validation_generator, test_generator, pred_generator, X_scaler, y_scaler


filename = Path('./MLP_model.joblib')

## Train model
def train(config, data):
  if (os.path.exists(filename)):
    return
  params = {
    'shuffle': True,
    'num_workers': 4,
    'n_feature':len(data['X_train'].columns),
    'n_output': len(data['y_train'].columns),
    'n_hidden_layers': 1,
    'batch_size': 64,
    'n_hidden': 512,
    'learning_rate': 1e-4,
    'epochs': 10,
    'dropout': 0.2,
    'log_nth': 1,
    'mode': 'train',
  }

  # Activate gpu optimization
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

  torch.manual_seed(42)

  model = Model(**params).to(device)
  optimizer = torch.optim.SGD(model.parameters(), lr=params['learning_rate'])
  criterion = nn.MSELoss(reduction='sum')

  model_dict, val_score = fit(data, model, device, params, config, optimizer, criterion)
  torch.save(model_dict, filename)


## Prediction
def predict(config, data):

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
  
  state = torch.load(filename)
  params = state['params']
  params['mode'] = 'predict'
  params['X_scaler'] = state['X_scaler']
  params['y_scaler'] = state['y_scaler']
  
  model = Model(**params).to(device)
  model.load_state_dict(state['model_dict'])

  pred_np, score = fit(data, model, device, params, config)
  pred_df = pd.DataFrame(index=data['X_test'].index)
  pred_df['MLP'] = pred_np[::config['window'], :].flatten()
  pred_df['True'] = data['y_test'].iloc[::config['window'], :].values.flatten()
  rmse = math.sqrt(metrics.mean_squared_error(pred_df['True'], pred_df['MLP']))
  r2 = metrics.r2_score(pred_df['True'], pred_df['MLP'])
  return pred_df, rmse, r2

## Pytorch Pipe 🐥
def fit(data, model, device, params, config, optimizer=None, criterion=None):
  training_generator, validation_generator, test_generator, pred_generator, X_scaler, y_scaler = get_datasets(data, **params)

  ## Run single training batch with backprop {loss}
  def runBatches(generator):
    losses = AverageMeter()

    for i, (X, y) in enumerate(generator):
      X, y = Variable(X, requires_grad=True).to(device), Variable(y).to(device)
      output = model.forward(X)
      loss = criterion(output, y)
      optimizer.zero_grad()
      loss.backward()
      # nn.utils.clip_grad_norm_(model.parameters(), params['clip'])
      optimizer.step()
      losses.update(loss.item())

    return losses.avg

  ## Run single prediction batch {y_true, y_pred}
  def predict(generator):
    model.eval()
    model.apply_dropout()
    y_trues = []
    y_preds = []
    
    for i, (X, y) in enumerate(generator):
      X, y = X.to(device), y.to(device)
      output = model.forward(X)
      y_trues = np.append(y_trues, y.cpu().numpy())
      y_preds = np.append(y_preds, output.detach().cpu().numpy())

    return np.array(y_trues), np.array(y_preds)

  ## Do Training
  if params['mode'] == 'train':
    start_time = datetime.datetime.now()
    train_scores = []
    val_scores = []

    best_model_dict = copy.deepcopy(model.state_dict())
    best_score = 999
    for epoch in range(params['epochs']):

      # Training
      model.train()
      train_score = runBatches(generator=training_generator)
      train_scores.append(train_score)

      # Validation
      model.eval()
      val_score = runBatches(generator=validation_generator)
      val_scores.append(val_score)

      # Keep the best model
      if val_score < best_score:
        best_score = val_score
        best_model_dict = copy.deepcopy(model.state_dict())

      time = (datetime.datetime.now() - start_time).total_seconds()
      
      if not epoch%params['log_nth']:
        print('e {e:<3} time: {t:<4.0f} train: {ts:<4.2f} val: {vs:<4.2f}'.format(e=epoch, t=time, ts=train_score, vs=val_score))

    # Test the trained model
    test_score = runBatches(generator=test_generator)
    trues, preds = predict(generator=pred_generator)

    # Return results, model and params for saving
    result_dict = {
      'model_dict': best_model_dict,
      'params': params,
      'train_scores': train_scores,
      'val_scores': val_scores,
      'X_scaler': X_scaler,
      'y_scaler': y_scaler,
    }
    return result_dict, best_score

  ## Do Predictions
  if params['mode'] == 'predict':
    trues, preds = predict(generator=pred_generator)
    score = math.sqrt(metrics.mean_squared_error(trues, preds))
    return y_scaler.inverse_transform(preds.reshape(-1, config['window'])), score

