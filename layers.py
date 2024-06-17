import numpy as np 
import utils

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        # TODO: return output
        pass

    def backward(self, output_gradient):
        # TODO: update weights_gradient and return input gradient
        pass

class Dense(Layer):
    def __init__(self, input_dims, units, activation ='Linear'):
        # input_shape = (batch_size, features)
        # input_dim = no of features
        self.units = units
        self.activation = utils.initiate_activation(activation)

        self.parameters = {
            "weights": np.random.randn(input_dims, units),
            "bias": np.random.randn(1, units)
        }
        
        self.gradients = {
            "weights": np.zeros_like(self.parameters["weights"]),
            "bias": np.zeros_like(self.parameters["bias"])
        }
    
    def forward(self, input):
        self.input = input
        out = np.dot(input, self.parameters["weights"]) + self.parameters["bias"]
        self.output = self.activation.forward(out)
        return self.output

    def backward(self, output_gradient):
        output_gradient = self.activation.backward(output_gradient)

        self.gradients["weights"] = np.dot(self.input.T, output_gradient)
        self.gradients["bias"] = np.mean(output_gradient, axis=0, keepdims= True)
        input_gradient = np.dot(output_gradient,self.parameters["weights"].T)
        return input_gradient


class RNN(Layer):
    def __init__(self, input_dim, units, activation = 'Tanh'):
        # input_shape = (batch_size, timesteps, features)
        # input_dim = no of features
        self.units = units
        self.activation = utils.initiate_activation(activation)

        self.parameters = {
            "weights": np.random.randn(input_dim, units),
            "recurrent_weights": np.random.randn(units, units),
            "bias": np.random.randn(1, units)*0.1
        }
       
        self.gradients = {
            "weights": np.zeros_like(self.parameters["weights"]),
            "recurrent_weights": np.zeros_like(self.parameters["recurrent_weights"]),
            "bias": np.zeros_like(self.parameters["bias"])
        }

    def forward(self, input):
        self.input = input
        batch_size, timesteps, features = input.shape

        self.states = np.zeros((batch_size, timesteps, self.units))
        prev_state = np.zeros((batch_size, self.units))
      
        for t in range(timesteps):
            state = self.activation.forward(
                np.dot(input[:,t,:],self.parameters["weights"]) + 
                np.dot(prev_state, self.parameters["recurrent_weights"]) + 
                self.parameters["bias"]
            )
            self.states[:,t,:] = self.activation.forward(state)
            prev_state = self.states[:,t,:]
    
        return self.states[:,-1,:]

    def backward(self, output_gradient):
        batch_size, timesteps, features = self.input.shape

        dWx = np.zeros_like(self.parameters["weights"])
        dWh = np.zeros_like(self.parameters["recurrent_weights"])
        db = np.zeros_like(self.parameters["bias"])
        dx = np.zeros_like(self.input)

        dL_dht = self.activation.backward(output_gradient)

        # Backpropagate through time
        for t in reversed(range(timesteps)):
            dWx += np.dot(self.input[:, t, :].T, dL_dht)
            db += np.sum(dL_dht, axis=0) / batch_size
            dx[:, t, :] = np.dot(dL_dht, self.parameters["weights"].T)

            if t>0:
                dWh += np.dot(self.states[:, t-1, :].T, dL_dht)
                dL_dht = np.dot(dL_dht, self.parameters["recurrent_weights"].T)

            value = 5
            dWx = np.clip(dWx, -value, value)
            dWh = np.clip(dWh, -value, value)
            db = np.clip(db, -value, value)
            dx = np.clip(dx, -value, value)
            dL_dht = np.clip(dL_dht, -value, value)

        self.gradients["weights"] = dWx
        self.gradients["recurrent_weights"] = dWh
        self.gradients["bias"] = db

        return dx
