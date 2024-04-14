import torch.nn as nn


class MLP(nn.Module):

    def __init__(self,
                 input_size=1,
                 hidden_sizes=[128, 64, 32],
                 output_size=1,
                 activation='ReLU'):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation = getattr(nn, activation)()

        # construct input layer
        self.layers = nn.ModuleList([nn.Linear(input_size, hidden_sizes[0])])

        # construct hidden layers
        layer_sizes = zip(hidden_sizes[:-1], hidden_sizes[1:])
        self.layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

        # construct output layer
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x
