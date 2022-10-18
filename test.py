import numpy as np

vector_weights = []

def create_nn(layer_sizes):
    n_layers = len(layer_sizes)
    for layer in range(n_layers-1):
        vector_weights[layer] = np.ones(layer_sizes[layer] + 1, layer_sizes[layer + 1])

def forward():
    pass


create_nn([3, 4, 3, 2])
print(vector_weights)