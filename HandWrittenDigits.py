from numba import jit
import numpy as np
from numpy.core import argmax
import tensorflow as tf


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print("====Shapes====")
print("X Train : ", x_train.shape)
print("Y Train : ", y_train.shape)
print("X Test : ", x_test.shape)
print("Y Train : ", y_train.shape)

x_train_inputs = []

for k in range(10000):
    x_train_tempo = []
    for i in range(28):
        for j in range(28):
            x_train_tempo.append(x_train[k][i][j])
    x_train_tempo = np.array(x_train_tempo)
    x_train_inputs.append(x_train_tempo)

y_train_inputs = y_train[:10000]

x_train_inputs = np.array(x_train_inputs)
y_train_inputs = np.array(y_train_inputs)


print("\nX Train Inputs : \n", x_train_inputs.shape)
print("\nY Train Inputs: \n", y_train_inputs.shape)

x_test_inputs = []

for k in range(1000):
    x_test_tempo = []
    for i in range(28):
        for j in range(28):
            x_test_tempo.append(x_test[k][i][j])
    x_test_tempo = np.array(x_test_tempo)
    x_test_inputs.append(x_test_tempo)

y_test_inputs = y_test[:1000]

x_test_inputs = np.array(x_test_inputs)
y_test_inputs = np.array(y_test_inputs)


print("\nX Test Inputs : ", x_test_inputs.shape)
print("Y Test Inputs: ", y_test_inputs.shape)

print("\n")

# === Layer Creation and Forward Propagation ==============================================================================

class layer:
    
    def __init__(self, n_inputs, n_neurons):

        self.weights = 0.1 * np.random.randn(n_neurons, n_inputs)  #Create a matrix of weight
        self.weights = self.weights.T   # Transpose the Matrix
        self.biases = np.zeros((1, n_neurons))  #Create a Matrix of Biases

    def forward(self, inputs):

        self.inputs = inputs 
        self.output = np.dot(inputs, self.weights) + self.biases # Calculate Inputs * Weights + Bias

    def backward(self, doutput):

        self.dinputs = np.dot(doutput, self.weights.T) # Calculate inputs derivative by multiplying output derivatives and weights
        self.dweights = np.dot(self.inputs.T, doutput) # Calculate weights derivative by multiplying output derivatives and Inputs
        self.dbiases = np.sum(doutput, axis=0, keepdims=True) # Calculate Biases derivative by adding output derivatives


class ActivationReLU:

    def forward(self, inputs):

        self.inputs = inputs

        self.output = np.maximum(0, inputs) # Keeping positive number and replacing negative by zero

        return self.output

    def backward(self, doutput):
        self.dinputs = doutput.copy() 

        self.dinputs[self.inputs <= 0] = 0 # Keeping in dinputs the values where the numbers with the same index are positive in inputs, if not replace by zero

class ActivationSoftmax:

    def forward(self, inputs):
        
        self.inputs = inputs

        self.expValues = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) # Calculate Exponential of inputs minus the max inputs to keeping value between -1 and 0
        self.probabilities = self.expValues / np.sum(self.expValues, axis=1, keepdims=True) # Division of each exponential value by the sum of all of them

        self.output = self.probabilities

        return self.output

    def backward(self, doutput):

        self.dinputs = np.empty_like(doutput) # Making an empty array with the same shape as doutput

        print("Enumerate : ", enumerate(zip(self.output, doutput)), "\n\n\n")

        for index, (single_output, single_doutput) in enumerate(zip(self.output, doutput)):  # Make an array wich is looking like that : [(0, (3, 6, 8, 4)), 3)]

            single_output = single_output.reshape(-1, 1)    # Turning 56 into [[56]]

            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)   ####### NEED TO EXPLAIN /!\ (need to understand shape of output and doutput)

            self.dinputs[index] = np.dot(jacobian_matrix, single_doutput)   ####### NEED TO EXPLAIN /!\


# === Manipulating Output ==================================================================================================

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class LossCategoricalCrossEntropy(Loss):

    def forward(self, y_input, y_true):
        
        sample_length = len(y_input) # Making a variable cointaining the sample length

        clipped_y_input = np.clip(y_input, 1e-7, 1-1e-7) # Took all values bellow 1e-7 and make them be 1e-7 and same with the value upper than 1-1e-7

        if len(y_true.shape) == 1 :

            predicted_value = clipped_y_input[range(sample_length), y_true] # Keeping value of the input at the same true value index 
            

        elif len(y_true.shape) == 2 :

            predicted_value = np.sum(clipped_y_input * y_true, axis=1) # Keeping value of the input at the same true value index

        self.loss = -np.log(predicted_value) # Calculating loss of the predicted value

        self.mean_loss = np.mean(self.loss) # Calculating mean loss

        return self.mean_loss

    def backward(self, doutput, y_true):

        samples = len(doutput) # Making a variable cointaining the length of the output's derivative

        labels = len(doutput[0]) # Making a variable cointaining the length of a row in the output's derivative

        if len(y_true)== 1 :
            y_true = np.eye(labels)[y_true] # Making a matrix with zeros and one whith the same index as y_true (it makes y_true beeing a 2D array)

        self.dinputs = -y_true / doutput # Calculating derivative of the inputs
        self.dinputs /= samples # Calculing derivative of the inputs

class AccuracyFunction:

    def forward(self, y_input, y_true):

        self.predictions=np.argmax(y_input, axis=1) # Stocked in an array indexes of the highest value
        
        if y_true.shape == 2:
            y_true = np.argmax(y_true, axis=1) # Stocked in an array indexes of the highest value (Converting in a y_true array)

        self.acc=np.mean(self.predictions==y_true) # Calculating the mean accuracy

        return self.acc

# === Optimizers ===========================================================================================================

class Optimizer_SGD():

    def __init__(self, learning_rate = 1., decay = 0., momentum = 0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_param(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations)) # Reducing decay the more iterations there is
        
    def update_param(self, layer):

        if self.momentum:

            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates

        else :
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases

        layer.weights += -self.learning_rate * layer.dweights # Modifying the weights
        layer.biases += -self.learning_rate * layer.dbiases # Modifying the biases

    def post_updates_params(self):
        self.iterations += 1 # Increase iterations by 1

class Optimizer_Adagrad():

    def __init__(self, learning_rate = 1., decay = 0., epsilon = 1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    def pre_update_param(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations)) # Reducing decay the more iterations there is
        
    def update_param(self, layer):

        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2

        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases +=  -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + + self.epsilon)

    def post_updates_params(self):
        self.iterations += 1 # Increase iterations by 1

class Optimizer_RMSprop():

    def __init__(self, learning_rate = 1., decay = 0., epsilon = 1e-7, rho = 0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    def pre_update_param(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations)) # Reducing decay the more iterations there is
        
    def update_param(self, layer):

        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache = self.rho * layer.weight_cache + (1-self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + (1-self.rho) * layer.dbiases**2

        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases +=  -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + + self.epsilon)

    def post_updates_params(self):
        self.iterations += 1 # Increase iterations by 1

class Optimizer_Adam():

    def __init__(self, learning_rate = 1e-3, decay = 0., epsilon = 1e-7, beta_1 = 0.9, beta_2 = 0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def pre_update_param(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations)) # Reducing decay the more iterations there is
        
    def update_param(self, layer):

        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_momentums = self.beta_1 * layer.weight_momentums +  (1-self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums +  (1-self.beta_1) * layer.dbiases

        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))


        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2

        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)

    def post_updates_params(self):
        self.iterations += 1 # Increase iterations by 1

# === Combined Class =======================================================================================================

class ActivationSoftmaxLoss():

    def __init__(self):

        self.activation = ActivationSoftmax() 
        self.loss = LossCategoricalCrossEntropy()

    def forward(self, inputs, yTrue):

        self.activation.forward(inputs)
        self.output = self.activation.output

        return self.loss.forward(self.output, yTrue)

    def backward(self, doutput, yTrue):

        samples = len(doutput)

        if len(yTrue.shape) == 2:
            yTrue = argmax(yTrue, axis=1)

        self.dinputs = doutput.copy()
        self.dinputs[range(samples), yTrue] -= 1
        self.dinputs /= samples
        

# === Running Code =========================================================================================================



ActivationHiddensLayers1 = ActivationReLU()
ActivationHiddensLayers2 = ActivationReLU()
ActivationOut = ActivationSoftmaxLoss()

optimizer = Optimizer_Adam(learning_rate = 0.02, decay=1e-5)

layer1 = layer(784, 350)
layer2 = layer(350, 100)
output = layer(100, 10)



for epoch in range(101):

    layer1.forward(x_train_inputs)
    ActivationHiddensLayers1.forward(layer1.output)
    layer2.forward(ActivationHiddensLayers1.output)
    ActivationHiddensLayers2.forward(layer2.output)
    output.forward(ActivationHiddensLayers2.output)

    loss = ActivationOut.forward(output.output, y_train_inputs)

    predictions = np.argmax(ActivationOut.output, axis=1)

    if len(y_train_inputs) == 2:
        y_train_inputs = argmax(y_train_inputs, axis=1)

    accuracy = np.mean(predictions==y_train_inputs)

    if not epoch % 10:
        print(f"Predictions : {predictions}")
        print(f'Epoch: {epoch}, ' +
        f'Acc: {accuracy:.3f}, ' +
        f'Loss: {loss:.3f}')
        print("\n")



    ActivationOut.backward(ActivationOut.output, y_train_inputs)
    output.backward(ActivationOut.dinputs)
    ActivationHiddensLayers2.backward(output.dinputs)
    layer2.backward(ActivationHiddensLayers2.dinputs)
    ActivationHiddensLayers1.backward(layer2.dinputs)
    layer1.backward(ActivationHiddensLayers1.dinputs)

    optimizer.pre_update_param()
    optimizer.update_param(layer1)
    optimizer.update_param(output)
    optimizer.post_updates_params()


layer1.forward(x_test_inputs)
ActivationHiddensLayers1.forward(layer1.output)
layer2.forward(ActivationHiddensLayers1.output)
ActivationHiddensLayers2.forward(layer2.output)
output.forward(ActivationHiddensLayers2.output)

loss = ActivationOut.forward(output.output, y_test_inputs)

predictions = np.argmax(ActivationOut.output, axis=1)

print("=== End Training =========================")
print(f"Predictions : {predictions[:30]}")
print(f"Y Test : {y_test[:30]}")
print(f'Epoch: Not Training, ' +
    f'Acc: {accuracy:.3f}, ' +
    f'Loss: {loss:.3f}')
print("\n")

