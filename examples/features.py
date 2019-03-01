import numpy as np
import pandas as pd

## Add Datetime Features
def add_datetime_features(data):
  data['hour'] = data.index.hour
  data['month'] = data.index.month
  data['year'] = data.index.year
  data['day_of_week'] = data.index.dayofweek
  data['day_of_month'] = data.index.day
  data['day_of_year'] = data.index.dayofyear

## Lagged Features
def add_hour_lag_features(data, labels=[], lags=[]):
  for label in labels:
    for lag in lags:
      data['feat_hourlag_{}_t-{}h'.format(label, lag)] = data[label].shift(lag)

## Moving Averages
def add_moving_average(data, labels=[], steps=[]):
  for label in labels:
    for step in steps:
      data['feat_ma_{}_p{}h'.format(label, step)] = data[label].rolling(step).mean()

## Exponential Moving Averages
def add_expenential_moving_average(data, labels=[], steps=[]):
  for label in labels:
    for step in steps:
      data['feat_ewm_{}_p{}h'.format(label, step)] = data[label].ewm(step).mean()

## Difference Features
def add_delta_feature(data, labels=[], steps=[]):
  for label in labels:
    for step in steps:
      data['feat_delta_{}_t-{}h'.format(label, step)] = data[label].diff(step)

## Deviation Features
def add_rolling_deviation(data, labels=[], steps=[]):
  for label in labels:
    for step in steps:
      data['feat_std_{}_p-{}h'.format(label, step)] = data[label].rolling(step).std()

## Variance Features
def add_rolling_variance(data, labels=[], steps=[]):
  for label in labels:
    for step in steps:
      data['feat_var_{}_p-{}h'.format(label, step)] = data[label].rolling(step).var()

## Wuhu
def add_features(data, labels):
  add_datetime_features(data)
  add_moving_average(data, labels=labels, steps=[3, 6, 12])
  add_hour_lag_features(data, labels=labels, lags=[1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12])
  add_delta_feature(data, labels=labels, steps=[1, 2, 3, 4, 5, 6, 12, 24, 48])
  add_rolling_deviation(data, labels=labels, steps=[2, 6, 12, 24])

  data = data.fillna(method='bfill')
  data = data.fillna(method='ffill')
  return data
