#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 15:08:21 2021

@author: 
"""

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from matplotlib import pyplot as plt

def get_features_targets(data):
  features = np.zeros((data.shape[0], 4))
  features[:,0] = data['u'] - data['g']
  features[:,1] = data['g'] - data['r']
  features[:,2] = data['r'] - data['i']
  features[:,3] = data['i'] - data['z']
  
  targets = data['redshift']
  return features, targets

# write a function that calculates the median of the differences
# between our predicted and actual values
def median_diff(predicted, actual):
  return np.median(np.abs(predicted - actual))

# write a function that splits the data into training and testing subsets
# trains the model and returns the prediction accuracy with median_diff
def validate_model(model, features, targets):
  # split the data into training and testing features and predictions
  split = features.shape[0]//2
  train_features = features[:split]
  train_targets = targets[:split]
  # train the model
  model.fit(train_features, train_targets)
  # get the predicted_redshifts
  test_features = features[split:]
  test_targets = targets[split:]
  predictions = model.predict(test_features)
  # use median_diff function to calculate the accuracy
  return median_diff(test_targets, predictions)


if __name__ == "__main__":
  data = np.load('sdss_galaxy_colors.npy')
  features, targets = get_features_targets(data)

  # initialize model
  dtr = DecisionTreeRegressor()

  # validate the model and print the med_diff
  diff = validate_model(dtr, features, targets)
  print('Median difference: {:f}'.format(diff))
      
  # Get a colour map
  cmap = plt.get_cmap('YlOrRd')
    
  # Define our colour indexes u-g and r-i
  x = data['u'] - data['g']
  y = data['r'] - data['i']
  # Make a redshift array
  redshift = data['redshift']
  # Create the plot with plt.scatter and plt.colorbar
  plot = plt.scatter(x=x, y=y, s=1, c=redshift, cmap=cmap, lw=0)
    
  cb = plt.colorbar(plot)
  cb.set_label('Redshift')
  # Define your axis labels and plot title
  plt.xlabel('Color index u-g')
  plt.ylabel('Color index r-i')
  plt.title('Redshift u-g vs r-i')
  # Set any axis limits
  plt.xlim(-0.5, 2.5)
  plt.ylim(-0.5, 1)
  plt.show()
