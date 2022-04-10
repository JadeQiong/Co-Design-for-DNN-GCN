from accelerator import Accelerator
from network import Network
import global_var
import random

class Population:
    def __init__(self, acc = Accelerator(), net = Network(), fit = 0, r_fit=0, c_fit=0):
        self.acc = acc
        self.net = net
        self.fit = fit
        self.r_fit = r_fit
        self.c_fit = c_fit

class GeneticAlgorithm:
    def __init__(self, acc, net, pop_num, iter_num, pf):
        self.accelerator = acc
        self.network = net
        self.pop_num = pop_num
        self.iter_num = iter_num
        self.p_factor = pf      # punish factor
        self.population = [Population() for i in range(self.pop_num)]
        self.best_pop = self.population[0]

    def initiate(self):
        for p in self.population:
            print(p.acc.topo)
            # hardware init
            p.acc.pe_num = random.randint(1, 10)
            p.acc.pe_numX = random.randint(1, 5)
            p.acc.pe_numY = random.randint(1, 5)
            p.acc.global_buf_size = random.random(10, 100)
            for i in p.acc.topo:
                i = random.randint(0,2)
            print(p.acc.topo)
            # software init
            # GG

    def keep_the_best(self):
        max_fitness = 0
        for p in self.population:
            cur_fitness = self.evaluate(p.acc, p.net)
            if max_fitness < cur_fitness:
                max_fitness = cur_fitness
                self.best_pop = p

    def select(self):
        return

    def run(self):
        fit = [0 for i in range(self.permutation_num)]
        return

    def cross(self):
        return

    def evaluate(self, h, net):
        # objective function
        t_comm = 0
        t_comp = 0

        # 乘法因子
        M = 0

        # constraints
        G_NUMBER = 10000
        area_thres = G_NUMBER
        energy_thres = G_NUMBER
        accuracy_thres = G_NUMBER
        area = h.pe_num * (global_var.a_other['router'] + global_var.a_other['sram'])
        energy = 0
        accuracy = 0

        # communication time
        mapping = [[] for i in range(net.num_Layers)]
        accumulate_pe_id = [0 for i in range(net.num_Layers)]
        mac_count = [0 for i in range(h.pe_num)]
        data_bit_width = 64

        for i in range(0, net.num_Layers):
            # ！这里不会算
            # 方案1：平均分配给每个PE
            tot_mac = net.layers[i][1] * net.layers[i][2]
            block_mac = tot_mac / h.pe_num
            # for each layer, we assign an accumulating PE(id = 0) for summing up partial sums
            accumulate_pe_id[i] = 0
            for j in range(0, h.pe_num):
                # mapping[i]: the i th layer is mapped to PE[0th ,1th, 2th, ... ]
                mac_count[j] = block_mac
                mapping[i].append(j)

        for i in range(0, net.num_Layers):
            max_hop = 0
            for j in mapping[i]:
                max_hop = max(max_hop, h.dis(accumulate_pe_id[i], j))
                bit_num = mac_count[j] * data_bit_width
                energy += global_var.e_trans * bit_num
            t_comm += max_hop * (global_var.t_trans + global_var.t_package + global_var.t_package)

        # computation time
        for i in range(0, net.num_Layers):
            max_time = 0
            for j in mapping[i]:
                max_time = max(max_time, mac_count[j] * global_var.t_mac[h.quantization[j]])
                energy += global_var.e_mac[h.quantization[j]] * mac_count[j]
            t_comp += max_time

        if energy > energy_thres or area > area_thres or accuracy < accuracy_thres:
            M = (energy-energy_thres)*(area-area_thres)*(accuracy-accuracy_thres)

        # TODO: return fitness function
        return 1/(t_comm+t_comp) * self.p_factor * M
