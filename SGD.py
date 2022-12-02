import NeuralNet
import numpy as np

class StochasticGradientDescent(NeuralNet.Backpropogater):
    def __init__(self, layer_sizes, activations, loss):
        super().__init__(layer_sizes,  activations, loss)

    def PropogateForward(self, input, weights):
        layers = [np.random.rand(size) for size in self.layer_sizes]
        layers[0] = input
        for idx, matrix in enumerate(weights):
            nonlinearized = self.activations[idx].func(input)
            input = matrix.T @ nonlinearized
            layers[idx+1] = input

        output = self.activations[-1].func(input)
        return layers, output

    def GetGradients(self, weights, biases, input, y):
        layers, output = self.PropogateForward(input, weights)

        nabla_nodes = [np.random.rand(size) for size in self.layer_sizes]
        nabla_weights = [np.random.rand(self.layer_sizes[i], self.layer_sizes[i+1]) for i in range(len(self.layer_sizes) - 1)]
        nabla_biases = [np.random.rand(self.layer_sizes[i+1],) for i in range(len(self.layer_sizes)-1)]

        nabla_nodes[-1] = self.loss.dfunc(output, y)


        target_layer_idx = len(self.layer_sizes) - 2

        while (target_layer_idx >= 0):
            

            x = layers[target_layer_idx]
            W = weights[target_layer_idx]
            dL_dy = nabla_nodes[target_layer_idx + 1] #(1, m)    dy_i/dx_j


            x, dL_dy = np.reshape(x, (-1, 1)), np.reshape(dL_dy, (1, -1)) #ensure shapes are correct

            activation = self.activations[target_layer_idx]


            #dL / dw_{i,j} = (dL/dy_j) * (dy_j/dw_{i,j}) = (dL/dy_j) * (f(x_i))
            dL_dw = activation.func(x) @ dL_dy

            nabla_weights[target_layer_idx] = dL_dw

            #dL / dx_i = sum_{j = 1...m} (dL/dy_j) * (dy_j/dx_i)
            # dy_j/dx_i = W * f'(x) [i,j]  <=== row i multiplied by element i
            # dL/dx_i = sum_{j = 1...m} (W * f'(x))[i,j] (dL_dy)[j, 1]      <=== treat dL_dy as column vector
            # dL/dx_i = ( (W*f'(x)) @ dL_dy )[i]

            #this will take the ith row of W and multiply it by the ith entry in f'(x). Took some playing to get this
            fp_x = activation.dfunc(x)


            dL_dx = (W.T * np.diag(fp_x)).T @ dL_dy.T
            nabla_nodes[target_layer_idx] = dL_dx


            # dL/db_i = dL/dy_i * dy_i/db_i = dL/dy_i * 1

            dL_db = np.copy(dL_dy)
            nabla_biases[target_layer_idx] = np.reshape(dL_db, (-1,))


            target_layer_idx -= 1

        return nabla_weights, nabla_biases
    
    def GetGradientsSingleParam(self, args):
        (weights, biases, input, y) = args
        return self.GetGradients(weights, biases, input, y)

class SigmoidActivation(NeuralNet.ActivationFunction):
    def __init__(self):
        pass
    def Sigmoid(self, x):
        return 1/(1 + np.exp(-1*x))
    def dSigmoid(self, x):
        return self.Sigmoid(x) * (1-self.Sigmoid(x))
    def func(self, x):
        return np.vectorize(self.Sigmoid)(x)
    def dfunc(self, x):
        return np.vectorize(self.dSigmoid)(x)

class LinearActivation(NeuralNet.ActivationFunction):
    def __init__(self):
        pass

    def func(self, x):
        return x
    
    def dfunc(self, x):
        return np.ones(x.shape)

class ReLUActivation(NeuralNet.ActivationFunction):
    def __init__(self):
        pass

    def func(self, x):
        return np.vectorize(lambda s: max(s, 0))(x)
    
    def dfunc(self, x):
        return np.vectorize(lambda s: 1 if s > 0 else 0)(x)

class MeanSquaredLoss(NeuralNet.LossFunction):
    def __init__(self):
        pass
    def func(self, a, y):
        return (a-y)**2
    def dfunc(self, a, y):
        return 2*(a-y)

class MeanAbsoluteLoss(NeuralNet.LossFunction):
    def __init__(self):
        pass
    def func(self, a, y):
        return np.abs(a-y)
    def dfunc(self, a, y):
        return -1 if a < y else 1