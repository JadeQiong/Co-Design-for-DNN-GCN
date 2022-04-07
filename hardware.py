import numpy as np
class Hardware(object):
    def __init__(self, PE_size, PE_numX, PE_numY, global_buf_size, pe_buffer_size):
        self.pe_size = PE_size
        self.pe_numX = PE_numX
        self.pe_numY = PE_numY
        self.global_buf_size = global_buf_size
        self.pe_buf_size = pe_buffer_size
        self.quantization = []
        self.pe_num = PE_numX*PE_numY
        # represent topology as a graph
        self.topo = np.zeros((self.pe_num, self.pe_num))
        for i in range(0, PE_numX-1):
            for j in range(0, PE_numY):
                id = i*PE_numY+j
                self.topo[id][id+1] = 1
        for j in range(0, PE_numY-1):
            for i in range(0, PE_numX):
                id = i*PE_numY+j
                self.topo[id][id] = 1
        # set quantization values
        for i in range(0, PE_numY*PE_numX):
            self.quantization.append(64)

    def print(self):
        print("PE size is " + str(self.pe_size))
        print("PE num is " + str(self.pe_numX * self.pe_numY))
        print("pe buffer size is " + str(self.pe_buf_size))
        print("global buffer size is " + str(self.global_buf_size))

h = Hardware(10,4,4,10,10)
h.pe_size = 177
h.print()
print(h.topo)

# TBC
# communication time
# computation time
