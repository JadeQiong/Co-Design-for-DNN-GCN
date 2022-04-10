import numpy as np
class Network:
    def __init__(self, layers = [("cov", 20, 20),("cov", 15, 15)]):
        self.num_Layers = len(layers)
        self.layers = layers
        self.layer_connection = np.zeros((self.num_Layers, self.num_Layers))
        for i in range(self.num_Layers-1):
            self.layer_connection[i][i+1] = 1
        self.accuracy = 0

    def print(self):
        print("number of layers is "+ str(self.num_Layers))
        print("layers ", end="")
        print(self.layers)
