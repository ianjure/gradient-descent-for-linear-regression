# Gradient Descent is an optimization algorithm for
# finding a local minimum of a differentiable function.

def stochastic_descend(x, y, w, b, lr):

    """
    Function: Gradient Descent for Linear Regression (Stochastic)

    Parameters:
    x - array of x values from y = wx + b
    y - array of true function (y = wx + b) outputs 
    w - initial random weight
    b - initial random bias
    lr - learning rate

    Optimize loss (Mean Square Error) for
    a linear function: y = wx + b
    """

    # Set the derivatives to zero first, we will update them in the learning step
    derivativeWeight = 0.0
    derivativeBias = 0.0

    # Iterate over the data to make the algorithm learn. (this is the learning step)
    """
    zip() function explained:

    a = ("John", "Charles", "Mike")
    b = ("Jenny", "Christy", "Monica", "Vicky")

    x = zip(a, b)

    output: (('John', 'Jenny'), ('Charles', 'Christy'), ('Mike', 'Monica'))
    """
    for xi, yi in zip(x,y):
        # Loss function to differentiate (MSE for a single observation | n=1): (1/1)(y - yhat)^2 or simply (y - yhat)^2
        # where yhat is wx + b | final function to differentiate: (y - (wx + b))^2

        # Derivative of the loss function with respect to weight | use chain rule
        """
        d/dw (y - (wx + b))^2
        d/db 2(y - (wx + b))(y - (wx + b))
        d/dw 2(y - (wx + b))(y - wx - b) | since y and b are treated as constants, their derivative is 0
        d/dw 2(y - (wx + b))(0 - x - 0) | since we will derive with respect to weight, wx = x
        d/dw 2(y - (wx + b))(-x)
        -2x(y - (wx + b))
        """
        derivativeWeight += -2 * xi * (yi - (w * xi + b))

        # Derivative of the loss function with respect to bias | use chain rule
        """
        d/db (y - (wx + b))^2
        d/db 2(y - (wx + b))(y - (wx + b))
        d/db 2(y - (wx + b))(y - wx - b) | since y and wx are treated as constants, their derivative is 0
        d/db 2(y - (wx + b))(0 - 0 - 1) | since we will derive with respect to bias, b = 1
        d/db 2(y - (wx + b))(-1)
        -2(y - (wx + b))
        """
        derivativeBias += -2 * (yi - (w * xi + b))

        # Add all the outputs of the derived functions to the variables we declared earlier
    
    # This is where we take a 'step' or update the parameters 'w' (weight) and 'b' (bias)
    # using the learning rate, 'N' or the amount of data, and the calculated derivative outputs
    # Gradient descent update formula: parameter - alpha(learning rate) * vector of partial derivatives of a loss function
    w = w - lr * derivativeWeight
    b = b - lr * derivativeBias

    return w, b

def batch_descend(x, y, w, b, lr):

    """
    Function: Gradient Descent for Linear Regression (Batch)

    Parameters:
    x - array of x values from y = wx + b
    y - array of true function (y = wx + b) outputs 
    w - initial random weight
    b - initial random bias
    lr - learning rate

    Optimize loss (Mean Square Error) for
    a linear function: y = wx + b
    """

    # Set the derivatives to zero first, we will update them in the learning step
    derivativeWeight = 0.0
    derivativeBias = 0.0

    # The amount of data we will feed the algorithm
    N = x.shape[0]

    # Iterate over the data to make the algorithm learn. (this is the learning step)
    """
    zip() function explained:

    a = ("John", "Charles", "Mike")
    b = ("Jenny", "Christy", "Monica", "Vicky")

    x = zip(a, b)

    output: (('John', 'Jenny'), ('Charles', 'Christy'), ('Mike', 'Monica'))
    """
    for xi, yi in zip(x,y):
        # Loss function to differentiate (MSE for the whole observation | n = amount of passed): (1/N)(y - yhat)^2
        # where yhat is wx + b | final function to differentiate: (1/N)(y - (wx + b))^2

        # Derivative of the loss function with respect to weight | use chain rule
        """
        d/dw (1/n)(y - (wx + b))^2
        Rule: d/dx (a * f) = a * d/dx (f)
        (1/n) d/dw 2(y - (wx + b))(y - (wx + b))
        (1/n) d/dw 2(y - (wx + b))(y - wx - b) | since y and b are treated as constants, their derivative is 0
        (1/n) d/dw 2(y - (wx + b))(0 - x - 0) | since we will derive with respect to weight, wx = x
        (1/n) d/dw 2(y - (wx + b))(-x)
        (2/n)(y - (wx + b))(-x)
        (2/n)(-x)(y - (wx + b))
        (-2/n)(x)(y - (wx + b))
        """
        derivativeWeight += (-2 / N) * xi * (yi - (w * xi + b))

        # Derivative of the loss function with respect to bias | use chain rule
        """
        d/db (1/n)(y - (wx + b))^2
        Rule: d/dx (a * f) = a * d/dx (f)
        (1/n) d/dw 2(y - (wx + b))(y - (wx + b))
        (1/n) d/db 2(y - (wx + b))(y - wx - b) | since y and wx are treated as constants, their derivative is 0
        (1/n) d/db 2(y - (wx + b))(0 - 0 - 1) | since we will derive with respect to bias, b = 1
        (1/n) d/db 2(y - (wx + b))(-1)
        (1/n) d/db -2(y - (wx + b))
        (-2/n)(y - (wx + b))
        """
        derivativeBias += (-2 / N) * (yi - (w * xi + b))

        # Add all the outputs of the derived functions to the variables we declared earlier
    
    # This is where we take a 'step' or update the parameters 'w' (weight) and 'b' (bias)
    # using the learning rate, 'N' or the amount of data, and the calculated derivative outputs
    # Gradient descent update formula: parameter - alpha(learning rate) * vector of partial derivatives of a loss function
    w = w - lr * derivativeWeight
    b = b - lr * derivativeBias

    return w, b