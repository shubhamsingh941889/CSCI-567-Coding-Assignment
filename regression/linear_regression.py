import numpy as np
import pandas as pd

############################################################################
# DO NOT MODIFY CODES ABOVE 
# DO NOT CHANGE THE INPUT AND OUTPUT FORMAT
############################################################################

###### Part 1.1 ######
def mean_square_error(w, X, y):
    """
    Compute the mean square error of a model parameter w on a test set X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing test features
    - y: A numpy array of shape (num_samples, ) containing test labels
    - w: a numpy array of shape (D, )
    Returns:
    - err: the mean square error
    """
    #####################################################
    # TODO 1: Fill in your code here                    #
    #####################################################
    c_x = np.size(X, 1)
    r_w = np.size(w, 0)
    if c_x == r_w:
      my_pred = np.dot(X, w)
    else:
      my_pred = np.dot(X.transpose(), w)
    sub = np.subtract(my_pred, y)
    err_sq = np.square(sub)
    err = np.mean(err_sq, dtype=np.float64)

    return err

###### Part 1.2 ######
def linear_regression_noreg(X, y):
  """
  Compute the weight parameter given X and y.
  Inputs:
  - X: A numpy array of shape (num_samples, D) containing features
  - y: A numpy array of shape (num_samples, ) containing labels
  Returns:
  - w: a numpy array of shape (D, )
  """
  #####################################################
  #	TODO 2: Fill in your code here                    #
  #####################################################		
  inv_trans_x_x = np.linalg.inv(np.dot(X.transpose(), X))
  x_trans_y = np.dot(X.transpose(), y)
  w = np.dot(inv_trans_x_x, x_trans_y)

  return w


###### Part 1.3 ######
def regularized_linear_regression(X, y, lambd):
    """
    Compute the weight parameter given X, y and lambda.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing features
    - y: A numpy array of shape (num_samples, ) containing labels
    - lambd: a float number specifying the regularization parameter
    Returns:
    - w: a numpy array of shape (D, )
    """
  #####################################################
  # TODO 4: Fill in your code here                    #
  #####################################################		
    identity_lambd = lambd * np.identity(np.size(X, 1))
    w = np.dot(np.linalg.inv(np.add(np.dot(X.transpose(), X), identity_lambd)), np.dot(X.transpose(), y))
    return w

###### Part 1.4 ######
def tune_lambda(Xtrain, ytrain, Xval, yval):
    """
    Find the best lambda value.
    Inputs:
    - Xtrain: A numpy array of shape (num_training_samples, D) containing training features
    - ytrain: A numpy array of shape (num_training_samples, ) containing training labels
    - Xval: A numpy array of shape (num_val_samples, D) containing validation features
    - yval: A numpy array of shape (num_val_samples, ) containing validation labels
    Returns:
    - bestlambda: the best lambda you find among 2^{-14}, 2^{-13}, ..., 2^{-1}, 1.
    """
    #####################################################
    # TODO 5: Fill in your code here                    #
    #####################################################		
    curr_min = float("inf")
    bestlambda = None
    for i in range(-14, 1):
        curr_pow = 2 ** i
        got_w = regularized_linear_regression(Xtrain, ytrain, curr_pow)
        got_err = mean_square_error(got_w, Xval, yval)
        if got_err < curr_min:
            curr_min = got_err
            bestlambda = curr_pow
    return bestlambda
    

###### Part 1.6 ######
def mapping_data(X, p):
    """
    Augment the data to [X, X^2, ..., X^p]
    Inputs:
    - X: A numpy array of shape (num_training_samples, D) containing training features
    - p: An integer that indicates the degree of the polynomial regression
    Returns:
    - X: The augmented dataset. You might find np.insert useful.
    """
    #####################################################
    # TODO 6: Fill in your code here                    #
    #####################################################		
    temp_X = X
    for power in range(2, p + 1):
        power_val = np.power(temp_X, power)
        X = np.concatenate((X, power_val), axis=1)
    return X

"""
NO MODIFICATIONS below this line.
You should only write your code in the above functions.
"""

