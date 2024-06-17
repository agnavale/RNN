import numpy as np

class SGD:
    def __init__(self,learning_rate = 0.1):
        self.learning_rate = learning_rate

    def update_parms(self, layer):
        for key in layer.parameters.keys():
            layer.parameters[key] -= self.learning_rate * layer.gradients[key]
        
        
class Momentum:
    def __init__(self, learning_rate = 0.1, beta = 0.9):
        self.learning_rate = learning_rate
        self.beta = beta

    def update_parms(self,layer):
        if not hasattr(layer,"velocity"):
            layer.velocity = {}
            for key in layer.parameters.keys():
                layer.velocity[key] = np.zeros(np.shape(layer.parameters[key]))

        for key in layer.parameters.keys():
                layer.velocity[key] = self.beta *  layer.velocity[key] + (1-self.beta)* layer.gradients[key]
                layer.parameters[key] -= self.learning_rate * layer.velocity[key]
     
class RMSprop:
    def __init__(self, learning_rate = 0.1, beta = 0.99):
        self.learning_rate = learning_rate
        self.beta = beta

    def update_parms(self,layer):
        if not hasattr(layer,"square"):
            layer.square = {}
            for key in layer.parameters.keys():
                layer.square[key] = np.zeros(np.shape(layer.parameters[key]))

        # Sw: weights_square Sb: bias_square
        for key in layer.parameters.keys():
            layer.square[key] =  np.clip(self.beta * layer.square[key] + (1-self.beta) * np.square(layer.gradients[key]), 1e-8, 1e+8)
            layer.parameters[key] -= self.learning_rate * layer.gradients[key] / np.sqrt(layer.square[key])

class Adam:
    def __init__(self, learning_rate = 0.1, beta1 = 0.9, beta2 = 0.99):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2

    def update_parms(self,layer):
        if not hasattr(layer, "velocity"):
            layer.velocity = {}
            layer.square = {}
            for key in layer.parameters.keys():
                layer.velocity[key] = np.zeros(np.shape(layer.parameters[key]))
                layer.square[key] = np.zeros(np.shape(layer.parameters[key]))
               

        for key in layer.parameters.keys():
            layer.velocity[key] = self.beta1 *  layer.velocity[key] + (1-self.beta1)* layer.gradients[key]
            layer.square[key] =  np.clip(self.beta2 * layer.square[key] + (1-self.beta2) * np.square(layer.gradients[key]), 1e-8, 1e+8)
            layer.parameters[key] -= self.learning_rate * layer.velocity[key] / np.sqrt(layer.square[key])
