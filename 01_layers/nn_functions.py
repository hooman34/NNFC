import numpy as np


def create_data(n, k):
    X = np.zeros((n*k, 2))  # data matrix (each row = single example)
    y = np.zeros(n*k, dtype='uint8')  # class labels
    for j in range(k):
        ix = range(n*j, n*(j+1))
        r = np.linspace(0.0, 1, n)  # radius
        t = np.linspace(j*4, (j+1)*4, n) + np.random.randn(n)*0.2  # theta
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = j
    return X, y


# Dense layer
class Layer_Dense:

    # Layer initialization
    def __init__(self, inputs, neurons, weight_regularizer_l1=0, weight_regularizer_l2=0,
                    bias_regularizer_l1=0, bias_regularizer_l2=0):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(inputs, neurons)
        self.biases = np.zeros((1, neurons))
        # set regularization strength
        # this is to set the strength of how much attention we are going to give, to regularization
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs
        # Calculate output values from input ones, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases

    # Backward pass
    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        
        # Gradients on regularization
        # L1 on weights
        if self.weight_regularizer_l1 > 0:
            dL1 = self.weights.copy()
            dL1[dL1 >= 0] = 1
            dL1[dL1 < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        # L2 on weights
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights
        # L1 on biases
        if self.bias_regularizer_l1 > 0:
            dL1 = self.biases.copy()
            dL1[dL1 >= 0] = 1
            dL1[dL1 < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        # L2 on biases
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

        # Gradient on values
        self.dvalues = np.dot(dvalues, self.weights.T)


# ReLU activation
class Activation_ReLU:

    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs
        # Calculate output values from input ones
        self.output = np.maximum(0, inputs)

    # Backward pass
    def backward(self, dvalues):
        # Since we need to modify original variable, 
        # let's make a copy of values first
        self.dvalues = dvalues.copy()

        # Zero gradient where input values were negative 
        self.dvalues[self.inputs <= 0] = 0 


# Softmax activation
class Activation_Softmax:

    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs

        # get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

    # Backward pass
    def backward(self, dvalues):
        self.dvalues = dvalues.copy()


# Cross-entropy loss
class Loss_CategoricalCrossentropy:

    # Forward pass
    def forward(self, y_pred, y_true):

        # Number of samples in a batch
        samples = y_pred.shape[0]

        # Probabilities for target values - only if categorical labels
        if len(y_true.shape) == 1:
            y_pred = y_pred[range(samples), y_true]

        # Losses
        negative_log_likelihoods = -np.log(y_pred)

        # Mask values - only for one-hot encoded labels
        if len(y_true.shape) == 2:
            negative_log_likelihoods *= y_true

        # Overall loss
        data_loss = np.sum(negative_log_likelihoods) / samples
        return data_loss

    # Backward pass
    def backward(self, dvalues, y_true):

        samples = dvalues.shape[0]

        self.dvalues = dvalues.copy()  # Copy so we can safely modify
        self.dvalues[range(samples), y_true] -= 1
        self.dvalues = self.dvalues / samples

class Optimizer_SGD:
    def __init__(self, learning_rate=1., decay = 0., momentum = 0.):
        """
        Enables learning rate decay and momentum.

        Args:
            learning_rate (float): learning rate
            decay (float): learning rate decay for each epoch
            momentum (float): how much to consider the global minimum
        """
        self.learning_rate = learning_rate
        self.current_learning_rate= learning_rate
        self.decay = decay
        self.momentum = momentum
        self.iterations = 0

    def pre_update_params(self):
        """
        Set learning rate. Do learning rate decay.
        """
        if self.decay:
            self.current_learning_rate = self.current_learning_rate * (1 / (1 + self.decay * self.iterations))

    def update_params(self, layer):
        """
        Update parameters, taking into account the momentums.
        """
        if not hasattr(layer, 'weight_momentums'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
        if self.momentum:
            # build weight updates
            weight_updates = (self.momentum * layer.weight_momentums) - (self.current_learning_rate * layer.dweights)
            layer.weight_momentums = weight_updates

            # build bias updates
            bias_updates = (self.momentum * layer.bias_momentums) - (self.current_learning_rate * layer.dbiases)
            layer.bias_momentums = bias_updates

        else:
            weight_updates = -(self.current_learning_rate * layer.dweights)
            bias_updates = -(self.current_learning_rate * layer.dbiases)

        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self):
        """
        Update after any parameter updates
        """
        self.iterations += 1

class Optimizer_Adagrad:
    def __init__(self, learning_rate=1., decay = 0., epsilon = 1e-7):
        """
        Enables learning rate decay and AdaGrad.

        Args:
            learning_rate (float): learning rate
            decay (float): learning rate decay for each epoch
            epsilon (float): committment to gradientes
        """
        self.learning_rate = learning_rate
        self.current_learning_rate= learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.iterations = 0

    def pre_update_params(self):
        """
        Set learning rate. Do learning rate decay.
        """
        if self.decay:
            self.current_learning_rate = self.current_learning_rate * (1 / (1 + self.decay * self.iterations))
        
    def update_params(self, layer):
        """
        Update parameters.
        For AdaGrad, it adjusts the learning rates for individual features.
        """

        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        
        # save in weight_cache
        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2

        # update weights
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)
        
    def post_update_params(self):
        """
        Update after any parameter updates
        """
        self.iterations += 1


class Optimizer_RMSprop:
    def __init__(self, learning_rate=1., decay = 0., epsilon = 1e-7, rho = 0.9):
        """
        Enables learning rate decay and AdaGrad.

        Args:
            learning_rate (float): learning rate
            decay (float): learning rate decay for each epoch
            epsilon (float): committment to gradientes
            rho (float): 
        """
        self.learning_rate = learning_rate
        self.current_learning_rate= learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.iterations = 0
        self.rho = rho

    def pre_update_params(self):
        """
        Set learning rate. Do learning rate decay.
        """
        if self.decay:
            self.current_learning_rate = self.current_learning_rate * (1 / (1 + self.decay * self.iterations))
        
    def update_params(self, layer):
        """
        Update parameters.
        Adjusts the learning rates for individual parameters.
        """

        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
                
        layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights ** 2
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases**2

        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)
                                

    def post_update_params(self):
        """
        Update after any parameter updates
        """
        self.iterations += 1


class Optimizer_Adam:
    def __init__(self, learning_rate=0.001, decay = 0., epsilon = 1e-7, beta_1 = 0.9, beta_2 = 0.999):
        """
        Enables learning rate decay and AdaGrad.

        Args:
            learning_rate (float): learning rate
            decay (float): learning rate decay for each epoch
            epsilon (float): committment to gradientes
            beta_1 (float): 
            beta_2 (float): 
        """
        self.learning_rate = learning_rate
        self.current_learning_rate= learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.iterations = 0
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def pre_update_params(self):
        """
        Set learning rate. Do learning rate decay.
        """
        if self.decay:
            self.current_learning_rate = self.current_learning_rate * (1. / (1. + self.decay * self.iterations))
        
    def update_params(self, layer):
        """
        Update parameters.
        Adjusts the learning rates for individual parameters.
        """

        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)
                
        layer.weight_cache = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases

        # Get corrected momentum
        # self.iteration is 0 at first pass
        # and we need to start with 1 here
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))

        # Update cache with squared current gradients
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2
        # Get corrected cache
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))


        # Vanilla SGD parameter update + normalization with square rooted cache
        layer.weights += -self.current_learning_rate * weight_momentums_corrected /\
                         (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected /\
                         (np.sqrt(bias_cache_corrected) + self.epsilon)

    def post_update_params(self):
        """
        Update after any parameter updates
        """
        self.iterations += 1

class Loss:
    def regularization_loss(self, layer):
        # init value
        regulartion_loss = 0
        # l1 regularization
        if layer.weight_regularizer_l1 > 0:
            regulartion_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
        if layer.bias_regularizer_l1 > 0:
            regulartion_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))
        # l2 regularization
        if layer.weight_regularizer_l2 > 0:
            regulartion_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)
        if layer.bias_regularizer_l2 > 0:
            regulartion_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)
        
        return regulartion_loss

# Cross-entropy loss
class Loss_CategoricalCrossentropy(Loss):

    # Forward pass
    def forward(self, y_pred, y_true):

        # Number of samples in a batch
        samples = y_pred.shape[0]

        # Probabilities for target values - only if categorical labels
        if len(y_true.shape) == 1:
            y_pred = y_pred[range(samples), y_true]

        # Losses
        negative_log_likelihoods = -np.log(y_pred)

        # Mask values - only for one-hot encoded labels
        if len(y_true.shape) == 2:
            negative_log_likelihoods *= y_true

        # Overall loss
        data_loss = np.sum(negative_log_likelihoods) / samples
        return data_loss

    # Backward pass
    def backward(self, dvalues, y_true):

        samples = dvalues.shape[0]

        dvalues = dvalues.copy()  # We need to modify variable directly, make a copy first then
        dvalues[range(samples), y_true] -= 1
        dvalues = dvalues / samples

        self.dvalues = dvalues

# Dropout
class Layer_Dropout:

    # Init
    def __init__(self, rate):
        self.rate = 1 - rate

    # Forward pass
    def forward(self, values):
        # Save input values
        self.input = values
       
        self.binary_mask = np.random.binomial(1, self.rate, size=values.shape) / self.rate
        # Apply mask to output values
        self.output = values * self.binary_mask

    # Backward pass
    def backward(self, dvalues):
        # Gradient on values
        self.dvalues = dvalues * self.binary_mask