from collections import OrderedDict
from scipy.special import expit
import numpy as np

class NeuralNetwork:
    """
    NeuralNetwork:{
    layer_id:{
                weights: np.array()
        }
    }
    
    Example:
    NeuralNetwork:{
    1:{
        'weights':[0.1,-0.54,0.32]
        }
    2:{
        'weights':[0.31,-0.344,0.41, 0.89]
        }    
    }
    
    """
    def __init__(self, units_per_layer, kernel_init, activation):
        self.units_per_layer = units_per_layer
        self.kernel_init = kernel_init
        self.activation = activation
        self.layers = OrderedDict()
        
        # initialize weights and biases 
        for layer_id, units in enumerate(self.units_per_layer[:-1]):
            prev_dims = self.units_per_layer[layer_id]
            num_units = self.units_per_layer[layer_id+1]
            layer_dims = (num_units, prev_dims)
            #layer_dims = (prev_dims, num_units, self.batch_size)
            
            self.layers[layer_id+1] = {}
            self.layers[layer_id+1]['weights'] = np.random.uniform(-1,1, layer_dims)
            # we can add biases later if we want
    
    def forward_pass(self, ipt_matrix):
        
        for layer_id, layer in self.layers.items():
            #print('layer number ->', layer_id)
            V = np.dot((ipt_matrix), (layer['weights'].T))
                
#                 print('multiplied matrices -> \n weights -> {}, \
#                 ipt_matrix ->{}'.format(layer['weights'].T.shape, ipt_matrix.shape))
#                 print('output matrix shape ', V.shape)
            y = self.activation_function(V,activation_type=self.activation[int(layer_id)])
            ipt_matrix = y
            
        return y
    
    
    @staticmethod
    def activation_function(V, activation_type='relu'):
        if activation_type == 'relu':
            return np.maximum(V, 0, V)
        elif activation_type == 'sigmoid':
            return expit(V)
        else:
            raise('activation type {} not found!'.format(activation_type))
            