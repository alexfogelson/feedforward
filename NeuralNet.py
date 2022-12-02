import numpy as np
from tqdm import tqdm
import multiprocessing
import warnings

#suppress warnings
warnings.filterwarnings('ignore')

class ActivationFunction:
    def __init__(self):
        pass
    def func(self, x): #vectorized!
        pass
    def dfunc(self, x):
        pass 

class LossFunction:
    def __init__(self, func, dfunc):
        pass
    def func(self, a,y):
        pass
    def dfunc(self, a, y):
        pass 

class Layer:
    def __init__(self, size, activation, bias = False):
        self.size = size 
        self.activation = activation
        self.bias = bias

class Backpropogater:
    def __init__(self, layer_sizes, activations, LossFunction):
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.loss = LossFunction()

    def GetGradients(self, weights, biases, input, y):
        pass

    def GetGradientsSingleParam(self, args):
        pass

class NeuralNet:
    
    def __init__(self, layers, PropogatorType, LossFunction, threshold = .5, scaling_factor = 1, n_workers = 1):
        self.scaling_factor = scaling_factor
        self.threshold = np.inf if threshold is None else threshold * scaling_factor * 10
        self.loss = LossFunction()
        self.n_workers = n_workers


        self.layer_sizes = [layer.size for layer in layers]
        self.activations = [layer.activation() for layer in layers]
        biases = [layer.bias for layer in layers]

        self.weights = [np.random.rand(self.layer_sizes[i], self.layer_sizes[i+1])*scaling_factor for i in range(len(layers) - 1)]

        self.biases = []
        for idx, bias in enumerate(biases):
            self.biases.append(np.random.rand(self.layer_sizes[idx+1], )*scaling_factor if bias else None) #biases are in the size of the next layer

        self.propogator = PropogatorType(self.layer_sizes, self.activations, LossFunction)

    def PropogateForward(self, input):
        #temporarily keep track o flayers
        layers = [np.random.rand(size) for size in self.layer_sizes]
        layers[0] = input
        for idx, matrix in enumerate(self.weights):
            nonlinearized = self.activations[idx].func(input)
            input = matrix.T @ nonlinearized 
            if (self.biases[idx] is not None):
                input += self.biases[idx]

            layers[idx+1] = input

        output = self.activations[-1].func(input)

        if (self.biases[-1] is not None):
            output += self.biases[-1]

        return layers, output
  
    def GetStep(self, batch_size = None):
        n = self.X.shape[0]
        cumulative_nabla_weight = [el*0 for el in self.weights]
        cumulative_nabla_biases = [None if el is None else el*0 for el in self.biases]

        if (batch_size is None):
            sample_choice_idxs = range(n)
        elif (batch_size < 1):
            sample_choice_idxs = np.random.choice(n, int(n*batch_size))
        else:
            sample_choice_idxs = np.random.choice(n, batch_size)


        with multiprocessing.Pool(self.n_workers) as p:
            training_pairs = [(self.weights, self.biases, self.X[i,:], self.y[i]) for i in sample_choice_idxs]

            results = p.map(self.propogator.GetGradientsSingleParam, training_pairs)

            for nabla_weights, nabla_biases in results:
                for j in range(len(cumulative_nabla_weight)):
                    cumulative_nabla_weight[j] += nabla_weights[j]
                    if (cumulative_nabla_biases[j] is not None):
                        cumulative_nabla_biases[j] += nabla_biases[j]

        return [el/n for el in cumulative_nabla_weight], [el if el is None else el/n for el in cumulative_nabla_biases]

    def GetLoss(self):
        n = self.X.shape[0]
        total_loss = 0
        for i in range(n):
            layers, output = self.PropogateForward(self.X[i,:]) 
            total_loss += self.loss.func(output, self.y[i])

        return total_loss/n
        
    def fit(self, X, y, epochs = 1000, alpha = .001, batch_size = 100):
        self.X = X
        self.y = y

        losses = []
        for i in tqdm(range(epochs)):
            weights_updates, bias_updates = self.GetStep(batch_size = batch_size)

            for i in range(len(self.weights)):
                nabla = np.minimum(weights_updates[i], self.threshold)
                self.weights[i] -= alpha*nabla
                
                if (bias_updates[i] is not None):
                    nabla = np.minimum(bias_updates[i], self.threshold)
                    self.biases[i] -= alpha*nabla

            losses.append(self.GetLoss())
        
        return losses

    def transform(self, X):
        n,m = X.shape
        ys = np.zeros(n)

        for i in range(n):
            _, ys[i] = self.PropogateForward(X[i,:])
        
        return ys

    def evaluate(self, a, y):
        n = a.shape[0]
        total_loss = 0
        for i in range(n):
            total_loss += self.loss.func(a[i], y[i])
        
        return total_loss/n





