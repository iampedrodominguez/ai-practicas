import numpy as np

vector_weights = []
activation_functions = []
n_layers = 0

#Crea la matriz de pesos de la red neuronal
def create_nn(layer_sizes):
    n_layers = len(layer_sizes)
    for layer in range(n_layers-1):
        vector_weights.append(np.ones((layer_sizes[layer] + 1, layer_sizes[layer + 1])))

#Forward para todos las capas de la red neuronal. 
#Recibe X: Input inicial, W: Matriz con los pesos de cada capa y la lista de funciones de activación. 
#Retorna Y: resultado de la red neuronal
def forward_process(X, W, activation_functions):
    Y = X
    #aumentando el bias
    Y= np.append(Y,[1])
    for layer in range(n_layers - 1):
        net = np.dot(Y, W[layer])
        Y = activation_functions[layer](net)
        # aumentando el bias
        Y= np.append(Y,[1])
    return Y

#Función de activación logística
#Definida como \frac{1}{1 + e^(-net)}
def logistic_function(net):
    return 1/(1+np.exp(-1*net))

#Guarda las funciones de activación para cada capa de la red neuronal
def fill_activation_functions(*_activation_functions):
    activation_functions = list(_activation_functions)

#Backward para las capas output-hidden de la red neuronal
def backward_hidden_ouput():
    pass

#Backward para las capas hidden-hidden de la red neuronal
def backward_hidden_hidden():
    pass

#Backward para todas las capas de la red neuronal
def backward_process():

    backward_hidden_ouput()
    backward_hidden_hidden()

create_nn([3, 4, 3, 2])
fill_activation_functions(logistic_function, logistic_function, logistic_function)
print(vector_weights)
print(activation_functions)