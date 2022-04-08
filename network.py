import numpy as np
class Network:
    def __init__(self, layers):
        self.num_Layers = len(layers)
        self.layers = layers
        self.layer_connection = np.zeros((self.num_Layers, self.num_Layers))
