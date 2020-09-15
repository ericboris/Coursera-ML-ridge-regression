#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import graphlab as gl
import numpy as np

##
## Week 1: Simple Linear Regression
##

# Return the intercept and slope of output with respect to input_feature
def simple_linear_regression(input_feature, output, verbose=False):    
    # Average all values of input and output into x and y
    # input_feature and output are assumed to be SArrays
    x = input_feature.mean()
    y = output.mean()
    
    n = input_feature.size()
    
    # Average x * y and x * x
    yx = (input_feature * output).sum() / n
    xx = (input_feature * input_feature).sum() / n
    
    # Find the slope, w1 and the y intercept, w0
    w1 = (x * y - yx) / (x * x - xx)
    w0 = y - (w1 * x)
     
    slope = w1
    intercept = w0
    
    if verbose is True:
        print("Simple Linear Regression")
        print("input feature: " + str(input_feature.head()))
        print("output: " + str(output.head()))
        print("intercept: %f" % intercept)
        print("slope: %f" % slope)
        
    return (intercept, slope)

# Predict the output y value given input_feature x
def get_regression_predictions(input_feature, intercept, slope, verbose=False):
    # Set parameters to mathematical notation
    # input_feature is assumed to be a float or an integer
    x = input_feature
    w0 = intercept
    w1 = slope
    
    # Solve for y_hat and convert back to programming notation
    y_hat = w0 + (w1 * x)
    prediction = y_hat
    
    if verbose is True:
        print("Get Regression Predictions")
        print("input feature: %f" % input_feature)
        print("intercept: %f" % intercept)
        print("slope: %f" % slope)
        print("prediction: %f" % prediction)
        
    return prediction

# Return the rss of a predicted feature and the true output. 
def get_residual_sum_of_squares_1(input_feature, output, intercept, slope, verbose=False):
    # input_feature and output are assumed to be SArrays
    x = input_feature
    y = output
    w0 = intercept
    w1 = slope
    
    y_hat = get_regression_predictions(x, w0, w1)
    r = y - y_hat
    
    rss = (r * r).sum()
    
    if verbose is True:
        print("Get Residual Sum of Squares 1")
        print("input feature: " + str(input_feature.head()))
        print("output: " + str(output.head()))
        print("intercept: %f" % intercept)
        print("slope: %f" % slope)
        print("rss: %f" % rss)
        
    return rss

# Return the inverse regression estimate, i.e. predict input given output
def inverse_regression_predictions(output, intercept, slope, verbose=False):
    # output is assumed to be a float or an integer
    y = output
    w0 = intercept
    w1 = slope
    
    x = (y - w0) / w1
    estimated_feature = x
    
    if verbose is True:
        print("Inverse Regression Predictions")
        print("output: %f" % output)
        print("intercept: %f" % intercept)
        print("slope: %f" % slope)
        print("estimated feature: %f" % estimated_feature)
    
    return estimated_feature

##
## Week 2: Multiple Linear Regression
##

# Modified version of get_residual_sum_of_squares_1
def get_residual_sum_of_squares(model, data, outcome, verbose=False):
    # Convert parameters to math notation
    h = model
    x = data
    y = outcome
    
    y_hat = h.predict(x)
    r = y - y_hat
    
    rss = (r * r).sum()
    
    if verbose is True:
        print("Get Residual Sum of Squares")
        print("model: \n%s\n" % str(model.get('coefficients')))
        print("predictions: \n%s\n" % str(y_hat.head()))
        print("outcome: \n%s\n" % str(outcome.head()))
        print("rss: \n%0.2f\n" % rss)
        
    return rss

# Return a numpy matrix of features from data_sframe with a constant column
def get_feature_matrix(data_sframe, features, verbose=False):
    # Convert to mathematical notation
    H = data_sframe
    w = features
    
    # Create a constant (all 1's) column on H
    H['constant'] = 1
    # Add a constant to features list
    w = ['constant'] + w
    # Populate an SFrame with w features from H
    w_sframe = gl.SFrame(H[w])
    # Convert it into a numpy matrix 
    W = w_sframe.to_numpy()
    
    # Convert back into programming notation
    feature_matrix = W
    
    if verbose is True:
        print("Get Feature Matrix")
        print("data sframe: \n" + str(data_sframe.head(1)))
        print("features: \n" + str(features) + "\n")
        print("feature matrix: \n" + str(feature_matrix))
        
    return feature_matrix

# Return a numpy array of output columns from data_sframe
def get_output_array(data_sframe, output, verbose=False):
    # Convert to mathematical notation
    H = data_sframe
    
    # Extract y desired columns from H
    y = H[output]
    
    # Convert to a numpy array for easier calculation
    output_array = y.to_numpy()
    
    if verbose is True:
        print("Get Output Array")
        print("data sframe: \n" + str(data_sframe.head(1)))
        print("output: \n" + str(output))
        print("output array \n" + str(y))
    
    return output_array

# Return a numpy matrix of features from data_sframe with a constant column
# And a numpy array of outputs from data_sframe
def get_numpy_data(data_sframe, features, output, verbose=False):
    feature_matrix = get_feature_matrix(data_sframe, features)
    
    output_array = get_output_array(data_sframe, output)
    
    if verbose is True:
        print("Get Numpy Data")
        print("data sframe: \n" + str(data_sframe.head(1)))
        print("features: \n" + str(features) + "\n")
        print("output: \n" + str(output) + "\n")
        print("feature matrix: \n" + str(feature_matrix) + "\n")
        print("output array \n" + str(output_array)+ "\n")
    
    return(feature_matrix, output_array)

# Predict the output of features based on given weights
def predict_output(feature_matrix, weights, verbose=False):
    # Convert to mathematical notation
    H = feature_matrix
    w = weights
    
    # Take the dot product of H and w
    y_hat = np.dot(H, w)
    predictions = y_hat
    
    if verbose is True:
        print("Get Numpy Data")
        print("feature matrix \n" + str(feature_matrix) + "\n")
        print("weights \n" + str(weights) + "\n")
        print ("predictions \n" + str(predictions) + "\n")
        
    return predictions

# Compute the derivative of the weight
# Given the value of the feature and the errors (for all data points)
def feature_derivative(errors, feature, verbose=False):
    # Convert to mathematical notation
    r = errors
    w = feature
    
    #Compute the derivative of weight with respect to r
    d = 2 * np.dot(r, w)
    derivative = d
    
    if verbose is True:
        print("Feature Derivative")
        print("errors: \n%s\n" % str(errors))
        print("feature: \n%s\n" % str(feature))
        print("derivative: \n%f\n" % derivative)
    
    return derivative

# Loop over all weights until they are minimized
def regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance, verbose=False):
    # Convert to mathematical notation
    H = feature_matrix
    y = output
    w = initial_weights
    eta = step_size
    eplison = tolerance
    
    converged = False # Used to stay in loop until convergence
    w = np.array(w) # Ensure that intial_weights are an array
    
    while not converged: # Loop over values while not within tolerance
        # Compute the predicted output of features based on weights
        y_hat = predict_output(H, w) 
        # And check how close those predictions were to the actual values
        r = y_hat - y
        
        # Initialize the starting sum of squares to zero
        gradient_sum_of_squares = 0
        
        # Update the weights for each feature
        for i in range(len(w)):
            # Compute the derivative for each weight
            der = feature_derivative(r, H[:, i])
            # Add the squared derivative to gradient_sum_of_squares to assess convergence
            gradient_sum_of_squares += (der ** 2)
            # Adjust the feature's weight, as derivative approaches 0 the change will decrease
            w[i] -= (eta * der)
            
        # Gradient magnitude will shrink over iterations until within tolerance
        gradient_magnitude = sqrt(gradient_sum_of_squares)
        
        if gradient_magnitude < tolerance:
            # When within tolerance, minimum is reached
            # Leave the while loop
            converged = True
    
    # Convert back to programming notation
    weights = w
    
    if verbose is True:
        print("Regression Gradient Descent")
        print("feature matrix: \n%s\n" % str(feature_matrix))
        print("output: \n%s\n" % str(output))
        print("initial weights: \n%s\n" % str(initial_weights))
        print("step size: \n%f\n" % step_size)
        print("tolerance: \n%f\n" % tolerance)
        print("weights: \n%s\n" % str(weights))
        
    # Return the minimized weights
    return(weights)


## Week 3: Assessing Performance


# Return an sframe with a feature column and
# Multiple feature columns raised to the degree power
def polynomial_sframe(feature, degree):
    # Initialize the sframe to return and create the first column
    poly_sframe = gl.SFrame()
    poly_sframe['power_1'] = feature
    # Ensure that degree is at least 1
    if degree < 1:
        degree = 1
    # Add feature columns up to degree-th power
    for power in range(2, degree+1): 
        name = 'power_' + str(power)
        poly_sframe[name] = feature.apply(lambda row: row ** power)
    return poly_sframe


## Week 4: 
