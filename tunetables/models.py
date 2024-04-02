import torch
import torch.nn as nn
import torch.nn.functional as F

class VMLP(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, activation_fn=nn.ReLU):
        super(VMLP, self).__init__()
        self.flatten = nn.Flatten()  # Flatten layer to handle multidimensional input
        self.layers = nn.ModuleList()

        # Assume input_size is either an int or a tuple/list for multidimensional inputs
        if isinstance(input_size, (tuple, list)):  # Calculate flattened input size if multidimensional
            flattened_size = 1
            for size in input_size:
                flattened_size *= size
        else:
            flattened_size = input_size

        # Input layer
        self.layers.append(nn.Linear(flattened_size, hidden_layers[0]))
        self.layers.append(activation_fn())

        # Hidden layers
        for i in range(1, len(hidden_layers)):
            self.layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
            self.layers.append(activation_fn())

        # Output layer
        self.layers.append(nn.Linear(hidden_layers[-1], output_size))

    def forward(self, x):
        x = self.flatten(x)  # Flatten the input
        for layer in self.layers:
            x = layer(x)
        return x

