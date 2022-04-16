import numpy as np

class Accelerator(object):
    def __init__(self, PE_numX = 4, PE_numY=4, PE_size = 10, tile_numX=3, tile_numY=3,global_buf_size=100, pe_topo=np.ones((0,0)), tile_topo=np.ones((0,0)),pe_buffer_size=100):
        self.pe_size = PE_size
        self.pe_numX = PE_numX
        self.pe_numY = PE_numY
        self.tile_numX = tile_numX
        self.tile_numY = tile_numY
        self.global_buf_size = global_buf_size
        self.pe_buf_size = pe_buffer_size
        self.quantization = []
        self.pe_num = PE_numX * PE_numY
        self.tile_num = tile_numX * tile_numY
        self.tile_topo = np.zeros((self.tile_num,self.tile_num))
        # represent topology as a graph
        self.pe_topo = np.zeros((self.pe_num, self.pe_num))
        if pe_topo.size != 0:
            self.pe_topo = pe_topo
        else:
            # print("init topo")
            for i in range(0, PE_numX - 1):
                for j in range(0, PE_numY):
                    id = i * PE_numY + j
                    self.pe_topo[id][id + PE_numY] = 1
                    self.pe_topo[id + PE_numY][id] = 1
            for j in range(0, PE_numY - 1):
                for i in range(0, PE_numX):
                    id = i * PE_numY + j
                    self.pe_topo[id][id + 1] = 1
                    self.pe_topo[id + 1][id] = 1
        # set quantization values
        for i in range(0, self.pe_num):
            self.quantization.append('64b')

        if tile_topo.size != 0:
            self.tile_topo = tile_topo
        else:
            # print("init topo")
            for m in range(0, tile_numX - 1):
                for n in range(0, tile_numY):
                    id2 = m * tile_numY + n
                    self.tile_topo[id2][id2 + tile_numY] = 1
                    self.tile_topo[id2 + tile_numY][id2] = 1
            for m in range(0, tile_numY - 1):
                for n in range(0, tile_numX):
                    id2 = m * tile_numY + n
                    self.tile_topo[id2][id2 + 1] = 1
                    self.tile_topo[id2 + 1][id2] = 1

    def print(self):
        print("PE size is " + str(self.pe_size))
        print("PE num is " + str(self.pe_numX * self.pe_numY))
        print("tile num is" +str(self.tile_numX * self.tile_numY))
        print("pe buffer size is " + str(self.pe_buf_size))
        print("global buffer size is " + str(self.global_buf_size))
        # print(self.topo)

    def pe_dis(self, id1, id2):
        maxn = 1e5
        num = self.pe_num
        dis = np.empty([num, num], dtype=int)
        if num != len(self.pe_topo):
            print(str(num) + "  GGGGGGGGGGGGGGGG " + str(len(self.pe_topo)))
        for i in range(0, num):
            for j in range(0, num):
                dis[i][j] = 1 if self.pe_topo[i][j] == 1 else maxn
        for k in range(0, num):
            for i in range(0, num):
                for j in range(0, num):
                    dis[i][j] = min(dis[i][j], dis[i][k]+dis[k][j])
        return dis[id1][id2]

    def tile_dis(self, id1, id2):
        maxn = 1e5
        num = self.pe_num
        dis = np.empty([num, num], dtype=int)
        if num != len(self.tile_topo):
            print(str(num) + "  GGGGGGGGGGGGGGGG " + str(len(self.pe_topo)))
        for i in range(0, num):
            for j in range(0, num):
                dis[i][j] = 1 if self.pe_topo[i][j] == 1 else maxn
        for k in range(0, num):
            for i in range(0, num):
                for j in range(0, num):
                    dis[i][j] = min(dis[i][j], dis[i][k] + dis[k][j])
        return dis[id1][id2]