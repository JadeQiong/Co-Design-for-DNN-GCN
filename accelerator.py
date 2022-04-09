import numpy as np

class Accelerator(object):
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
                self.topo[id][id+PE_numY] = 1
                self.topo[id+PE_numY][id] = 1
        for j in range(0, PE_numY-1):
            for i in range(0, PE_numX):
                id = i*PE_numY+j
                self.topo[id][id+1] = 1
                self.topo[id+1][id] = 1
        # set quantization values
        for i in range(0, self.pe_num):
            self.quantization.append('64b')

    def print(self):
        print("PE size is " + str(self.pe_size))
        print("PE num is " + str(self.pe_numX * self.pe_numY))
        print("pe buffer size is " + str(self.pe_buf_size))
        print("global buffer size is " + str(self.global_buf_size))
        # print(self.topo)

    def dis(self, id1, id2):
        maxn = 1e5
        num = self.pe_num
        dis = np.empty([num, num], dtype=int)
        for i in range(0, num):
            for j in range(0, num):
                dis[i][j] = 1 if self.topo[i][j] == 1 else maxn
        for k in range(0, num):
            for i in range(0, num):
                for j in range(0, num):
                    dis[i][j] = min(dis[i][j], dis[i][k]+dis[k][j])
        return dis[id1][id2]
