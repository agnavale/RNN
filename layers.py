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

class LSTM(Layer):
    def __init__(self, input_dim, units, activation='Tanh', recurrent_activation='Sigmoid'):
        self.units = units
        self.activation = utils.initiate_activation(activation)
        self.recurrent_activation = utils.initiate_activation(recurrent_activation)

        self.parameters = {
            "Wf": np.random.randn(input_dim, units),
            "Wi": np.random.randn(input_dim, units),
            "Wc": np.random.randn(input_dim, units),
            "Wo": np.random.randn(input_dim, units),
            "Uf": np.random.randn(units, units),
            "Ui": np.random.randn(units, units),
            "Uc": np.random.randn(units, units),
            "Uo": np.random.randn(units, units),
            "bf": np.zeros((1, units)),
            "bi": np.zeros((1, units)),
            "bc": np.zeros((1, units)),
            "bo": np.zeros((1, units)),
        }

        self.gradients = {k: np.zeros_like(v) for k, v in self.parameters.items()}

    def forward(self, input):
        self.input = input
        batch_size, timesteps, features = input.shape

        self.h_states = np.zeros((batch_size, timesteps, self.units))
        self.c_states = np.zeros((batch_size, timesteps, self.units))
        prev_h = np.zeros((batch_size, self.units))
        prev_c = np.zeros((batch_size, self.units))

        for t in range(timesteps):
            x_t = input[:, t, :]

            f_t = self.recurrent_activation.forward(
                np.dot(x_t, self.parameters["Wf"]) + np.dot(prev_h, self.parameters["Uf"]) + self.parameters["bf"]
            )
            i_t = self.recurrent_activation.forward(
                np.dot(x_t, self.parameters["Wi"]) + np.dot(prev_h, self.parameters["Ui"]) + self.parameters["bi"]
            )
            c_tilde_t = self.activation.forward(
                np.dot(x_t, self.parameters["Wc"]) + np.dot(prev_h, self.parameters["Uc"]) + self.parameters["bc"]
            )
            c_t = f_t * prev_c + i_t * c_tilde_t
            o_t = self.recurrent_activation.forward(
                np.dot(x_t, self.parameters["Wo"]) + np.dot(prev_h, self.parameters["Uo"]) + self.parameters["bo"]
            )
            h_t = o_t * self.activation.forward(c_t)

            self.h_states[:, t, :] = h_t
            self.c_states[:, t, :] = c_t

            prev_h = h_t
            prev_c = c_t

        return self.h_states[:, -1, :]

    def backward(self, output_gradient):
        batch_size, timesteps, features = self.input.shape

        dWf, dWi, dWc, dWo = np.zeros_like(self.parameters["Wf"]), np.zeros_like(self.parameters["Wi"]), np.zeros_like(self.parameters["Wc"]), np.zeros_like(self.parameters["Wo"])
        dUf, dUi, dUc, dUo = np.zeros_like(self.parameters["Uf"]), np.zeros_like(self.parameters["Ui"]), np.zeros_like(self.parameters["Uc"]), np.zeros_like(self.parameters["Uo"])
        dbf, dbi, dbc, dbo = np.zeros_like(self.parameters["bf"]), np.zeros_like(self.parameters["bi"]), np.zeros_like(self.parameters["bc"]), np.zeros_like(self.parameters["bo"])
        dx = np.zeros_like(self.input)

        dL_dht = self.activation.backward(output_gradient)
        dL_dct = np.zeros_like(dL_dht)

        for t in reversed(range(timesteps)):
            x_t = self.input[:, t, :]
            h_t = self.h_states[:, t, :]
            c_t = self.c_states[:, t, :]

            prev_h = self.h_states[:, t-1, :] if t > 0 else np.zeros_like(h_t)
            prev_c = self.c_states[:, t-1, :] if t > 0 else np.zeros_like(c_t)

            o_t = self.recurrent_activation.forward(
                np.dot(x_t, self.parameters["Wo"]) + np.dot(prev_h, self.parameters["Uo"]) + self.parameters["bo"]
            )
            f_t = self.recurrent_activation.forward(
                np.dot(x_t, self.parameters["Wf"]) + np.dot(prev_h, self.parameters["Uf"]) + self.parameters["bf"]
            )
            i_t = self.recurrent_activation.forward(
                np.dot(x_t, self.parameters["Wi"]) + np.dot(prev_h, self.parameters["Ui"]) + self.parameters["bi"]
            )
            c_tilde_t = self.activation.forward(
                np.dot(x_t, self.parameters["Wc"]) + np.dot(prev_h, self.parameters["Uc"]) + self.parameters["bc"]
            )

            dL_dct += dL_dht * o_t * self.activation.backward(c_t)
            dL_dht = dL_dht * self.activation.forward(c_t) * self.recurrent_activation.backward(o_t)

            dL_dct_tilde = dL_dct * i_t * self.activation.backward(c_tilde_t)
            dL_dft = dL_dct * prev_c * self.recurrent_activation.backward(f_t)
            dL_dit = dL_dct * c_tilde_t * self.recurrent_activation.backward(i_t)
            dL_dct = dL_dct * f_t

            dWc += np.dot(x_t.T, dL_dct_tilde)
            dUc += np.dot(prev_h.T, dL_dct_tilde)
            dbc += np.sum(dL_dct_tilde, axis=0) / batch_size

            dWf += np.dot(x_t.T, dL_dft)
            dUf += np.dot(prev_h.T, dL_dft)
            dbf += np.sum(dL_dft, axis=0) / batch_size

            dWi += np.dot(x_t.T, dL_dit)
            dUi += np.dot(prev_h.T, dL_dit)
            dbi += np.sum(dL_dit, axis=0) / batch_size

            dWo += np.dot(x_t.T, dL_dht)
            dUo += np.dot(prev_h.T, dL_dht)
            dbo += np.sum(dL_dht, axis=0) / batch_size

            dx[:, t, :] = np.dot(dL_dht, self.parameters["Wo"].T) + np.dot(dL_dct_tilde, self.parameters["Wc"].T) + np.dot(dL_dft, self.parameters["Wf"].T) + np.dot(dL_dit, self.parameters["Wi"].T)

            value = 5
            dWf = np.clip(dWf, -value, value)
            dUf = np.clip(dUf, -value, value)
            dbf = np.clip(dbf, -value, value)

            dWi = np.clip(dWi, -value, value)
            dUi = np.clip(dUi, -value, value)
            dbi = np.clip(dbi, -value, value)

            dWc = np.clip(dWc, -value, value)
            dUc = np.clip(dUc, -value, value)
            dbc = np.clip(dbc, -value, value)

            dWo = np.clip(dWo, -value, value)
            dUo = np.clip(dUo, -value, value)
            dbo = np.clip(dbo, -value, value)

            dx = np.clip(dx, -value, value)

        self.gradients["Wf"] = dWf
        self.gradients["Uf"] = dUf
        self.gradients["bf"] = dbf

        self.gradients["Wi"] = dWi
        self.gradients["Ui"] = dUi
        self.gradients["bi"] = dbi

        self.gradients["Wc"] = dWc
        self.gradients["Uc"] = dUc
        self.gradients["bc"] = dbc

        self.gradients["Wo"] = dWo
        self.gradients["Uo"] = dUo
        self.gradients["bo"] = dbo

        return dx
        

    
    
