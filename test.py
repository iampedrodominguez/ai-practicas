import numpy as np

vector_weights = []
activation_functions = []
n_layers = 0

def create_nn(layer_sizes):
    n_layers = len(layer_sizes)
    for layer in range(n_layers-1):
        vector_weights.append(np.ones((layer_sizes[layer] + 1, layer_sizes[layer + 1])))

#Forward para todos las capas de la red neuronal. 
# Recibe X: Input inicial, W: Matriz con los pesos de cada capa y la lista de funciones de activación. 
# Retorna Y: resultado de la red neuronal
def forward_layer(X, W, activation_functions):
    Y = X

    for layer in range(n_layers - 1):
        net = np.dot(Y, W[layer])
        Y = activation_functions[layer](net)

    return Y


create_nn([3, 4, 3, 2])
print(vector_weights)