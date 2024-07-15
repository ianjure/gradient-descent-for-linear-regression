import numpy as np
from gradientDescent import stochastic_descend, batch_descend

# INITIALIZE PARAMETERS
x = np.random.randn(10,1)
tw = 5 # -- true weight that we want to predict
tb = 10 # -- true bias that we want to predict
y = tw*x + tb # -- Linear Regression Function: y = mx + b


# Variables that we want to update, after using gradient descent, this should be equal to the
# true weight and bias that we set earlier
w = 0.0 
b = 0.0

# HYPERPARAMETER
learning_rate = 0.01

# INITIALIZE VARIABLES
epochs = 500

print(f'STOCHASTIC GRADIENT DESCENT')

# ITERATIVELY MAKE UPDATES, 'w' and 'b' should improve and move closer to 'tw' and 'tw' for each epoch
for epoch in range(epochs):
    w, b = stochastic_descend(x=x, y=y, w=w, b=b, lr=learning_rate) # -- USING STOCHASTIC GRADIENT DESCENT

    # VALIDATION
    # calculate the predictions, compared to 'y' function earlier for each x
    # yhat will return a numpy array of calculated yhat
    yhat = w*x + b # Linear Regression Function: y = mx + b

    # calculate the loss for each yhat, how far off are the predictions (yhat) to the true value (y)
    # use MSE (Mean Squared Error) since this is what we used in the gradient descent function
    # MSE = (1/n) summation of (y - yhat)^2
    loss = np.sum((y - yhat)**2, axis=0) / x.shape[0]

    # SHOW RESULT EVERY 50 EPOCH
    if epoch % 50 == 0:
        print(f'{epoch} | LOSS: {loss} | PARAMETERS: w:{w}, b:{b}')

print(f'TRUE PARAMETERS: w:{tw}, b:{tb} | PREDICTED PARAMETERS: w:{w}, b:{b}')

# ---------------------------------------------------------------------------------------------------- #

print(f'BATCH GRADIENT DESCENT')

# ITERATIVELY MAKE UPDATES, 'w' and 'b' should improve and move closer to 'tw' and 'tw' for each epoch
for epoch in range(epochs):
    w, b = batch_descend(x=x, y=y, w=w, b=b, lr=learning_rate) # -- USING BATCH GRADIENT DESCENT

    # VALIDATION
    # calculate the predictions, compared to 'y' function earlier for each x
    # yhat will return a numpy array of calculated yhat
    yhat = w*x + b # Linear Regression Function: y = mx + b

    # calculate the loss for each yhat, how far off are the predictions (yhat) to the true value (y)
    # use MSE (Mean Squared Error) since this is what we used in the gradient descent function
    # MSE = (1/n) summation of (y - yhat)^2
    loss = np.sum((y - yhat)**2, axis=0) / x.shape[0]

    # SHOW RESULT EVERY 50 EPOCH
    if epoch % 50 == 0:
        print(f'{epoch} | LOSS: {loss} | PARAMETERS: w:{w}, b:{b}')

print(f'TRUE PARAMETERS: w:{tw}, b:{tb} | PREDICTED PARAMETERS: w:{w}, b:{b}')