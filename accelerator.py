import numpy as np
import random
import global_var


def get_dis(dis, num, id1, id2):
    for k in range(num):
        for i in range(num):
            for j in range(num):
                dis[i][j] = min(dis[i][j], dis[i][k] + dis[k][j])
    return dis[id1][id2]

# 带权topo初始化
def initiate_topo(topo, x, y):
    for i in range(x - 1):
        for j in range(y):
            topo_id = i * y + j
            weight = random.randint(1, global_var.max_bandwidth)
            topo[topo_id][topo_id + y] = weight
            topo[topo_id + y][topo_id] = weight
    for j in range(y - 1):
        for i in range(x):
            topo_id = i * y + j
            weight = random.randint(1, global_var.max_bandwidth)
            topo[topo_id][topo_id + 1] = weight
            topo[topo_id + 1][topo_id] = weight


class Accelerator(object):

    def __init__(self, uselist=False, acc_gene=[],
                 PE_numX=12, PE_numY=12, tile_numX=3, tile_numY=3, PE_size=10,
                 global_buf_size=100, pe_topo=np.ones((0, 0)), tile_topo=np.ones((0, 0)), pe_buffer_size=100):
        if uselist is False:
            self.pe_size = PE_size
            self.pe_numX = PE_numX
            self.pe_numY = PE_numY
            self.tile_numX = tile_numX
            self.tile_numY = tile_numY
            self.global_buf_size = global_buf_size
            self.pe_buf_size = pe_buffer_size
            self.quantization = []
            tile_num = tile_numX * tile_numY
            self.tile_topo = np.zeros((tile_num, tile_num))
            # represent topology as a graph
            pe_num = self.pe_numX * self.pe_numY
            self.pe_topo = np.zeros((pe_num, pe_num))
            if tile_topo.size != 0:
                self.tile_topo = tile_topo
            else:
                initiate_topo(self.tile_topo, self.tile_numX, self.tile_numY)
            if pe_topo.size != 0:
                self.pe_topo = pe_topo
            else:
                initiate_topo(self.pe_topo, self.pe_numX, self.pe_numY)

        else:
            # 直接用acc_gene赋值，不用初始化
            self.pe_numX = acc_gene[0]
            self.pe_numY = acc_gene[1]
            self.tile_numX = acc_gene[2]
            self.tile_numY = acc_gene[3]
            self.pe_size = acc_gene[4]
            self.global_buf_size = acc_gene[5]
            self.pe_topo = acc_gene[6]
            self.tile_topo = acc_gene[7]
            self.quantization = []
            # set quantization values
            for i in range(0, self.pe_numY * self.pe_numY):
                self.quantization.append('64b')

    def tile_topo_dis(self, id1, id2):
        maxn = 1e5
        num = len(self.tile_topo)
        dis = np.empty([num, num], dtype=int)
        for i in range(num):
            for j in range(num):
                dis[i][j] = 1 if self.tile_topo[i][j] == 1 else maxn
        return get_dis(dis, num, id1, id2)

    def tile_id(self, pe_id):
        idx = pe_id / self.pe_numY
        idy = pe_id % self.pe_numY
        tile_x = idx / self.tile_numX
        tile_y = idy / self.tile_numY
        tile_id = tile_x * self.tile_numY + tile_y
        return tile_id

    def pe_topo_dis(self, id1, id2):
        num = len(self.pe_topo)
        dis = np.empty([num, num], dtype=int)
        for i in range(num):
            for j in range(num):
                tile_i = self.tile_id(i)
                tile_j = self.tile_id(j)
                # the same tile
                if tile_i == tile_j:
                    dis[i][j] = self.pe_topo[i][j]
                else:
                    dis[i][j] = self.tile_topo_dis(tile_i, tile_j) * global_var.inter_tile_cost
        return get_dis(dis, num, id1, id2)

    def print(self):
        print("PE size is " + str(self.pe_size))
        print("PE num is " + str(self.pe_numX * self.pe_numY))
        print("tile num is" + str(self.tile_numX * self.tile_numY))
        print("pe buffer size is " + str(self.pe_buf_size))
        print("global buffer size is " + str(self.global_buf_size))
        # print(self.topo)

