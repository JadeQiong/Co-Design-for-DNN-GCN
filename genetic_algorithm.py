from accelerator import Accelerator
from network import Network
import genetic_algorithm_var
import global_var
import random


class Population:
    def __init__(self, acc=Accelerator(), net=Network(), fit=0, r_fit=0, c_fit=0):
        self.acc_gene = [acc.pe_num, acc.pe_size, acc.pe_numX, acc.pe_numY, acc.global_buf_size, acc.topo]
        self.net_gene = [net.num_Layers, net.layers, net.layer_connection]
        self.fit = fit
        self.r_fit = r_fit
        self.c_fit = c_fit


def encode_int(num):
    s = []
    while int(num) != 0:
        s.append(num % 2)
        num = int(num/2)
    dif = genetic_algorithm_var.int_code_len-len(s)
    for i in range(0, dif):
        s.append(0)
    return s


def decode_int(s):
    num = 0
    for i in range(0, len(s)):
        if s[i] == 1:
            num += pow(2, i)
    return num


def encode_float(num):
    return


def decode_float(s):
    return


def encode_topo(topo):
    s = []
    return s


def decode_topo(s):
    topo = 1
    return topo


def test_int_encoding():
    for i in range(100):
        if (decode_int(encode_int(i)) != i):
            print(i)
            print(encode_int(i))
            print(decode_int(encode_int(i)))
            print("test failed")

class GeneticAlgorithm:

    def __init__(self, pop_num=10, iter_num=10, gen_num=10, pf=1):
        self.pop_num = pop_num
        self.iter_num = iter_num
        self.gen_num = gen_num
        self.p_factor = pf      # punish factor
        self.population = [Population() for i in range(self.pop_num)]
        self.next_population = [Population() for i in range(self.pop_num)]
        self.best_pop = self.population[0]

    def run(self):
        self.initiate()
        self.evaluate()
        self.keep_the_best()
        # self.crossover()
        for attr_id in range(0, len(genetic_algorithm_var.acc_gene_type)):
            attr = genetic_algorithm_var.acc_gene_type[attr_id]
            self.crossover(attr_id, attr)
        gen = 0
        for cur_iter in range(0, self.iter_num):
            while gen < self.gen_num:
                gen += 1
                self.select()
                for attr_id in range(0, len(genetic_algorithm_var.acc_gene_type)):
                    attr = genetic_algorithm_var.acc_gene_type[attr_id]
                    self.crossover(attr_id, attr)
                self.mutate()
                self.evaluate()
                # self.keep_the_best()
                self.elitist()
                print("best in each iteration----------------")
                print(self.best_pop.acc_gene[0:4])
                print(self.best_pop.net_gene[0:4])
        return

    def initiate(self):
        for p in self.population:
            # hardware init
            for j in range(0, len(genetic_algorithm_var.acc_gene_type)):
                gene = genetic_algorithm_var.acc_gene_type[j]
                if gene == "pe_num":
                    p.acc_gene[j] = random.randint(1, 10)
                elif gene == "pe_size":
                    p.acc_gene[j] = random.randint(1, 10)
                elif gene == "pe_numX":
                    p.acc_gene[j] = random.randint(1, 5)
                elif gene == "pe_numY":
                    p.acc_gene[j] = random.randint(1, 5)
                elif gene == "global_buffer_size":
                    p.acc_gene[j] = random.randint(1, 5)
                elif gene == "topo":
                    pass
                    # print(p.acc_gene[j])
                    # for i in p.acc_gene[j]:
                    #     i = random.randint(0, 2)
            # print(p.acc.topo)
            # software init
            # GG

    def keep_the_best(self):
        for p in self.population:
            if self.best_pop.fit < p.fit:
                self.best_pop = p

    def select(self):
        fit_sum = 0
        # 轮盘赌
        for p in self.population:
            fit_sum += p.fit
        for p in self.population:
            p.r_fit = p.fit/fit_sum
        for i in range(0, len(self.population)):
            if i == 0:
                self.population[i].c_fit = self.population[i].fit
            else:
                self.population[i].c_fit = self.population[i-1].c_fit + self.population[i].fit
        # generate random probability
        for i in range(0, len(self.population)):
            probability = random.uniform(0, 1)
            if probability < self.population[0].c_fit:
                self.next_population[i] = self.population[i]
            else:
                for j in range(0, len(self.population)-1):
                    if self.population[j].c_fit <= probability < self.population[j + 1].c_fit:
                        self.next_population[i] = self.population[j+1]
        # update population
        self.population = self.next_population
        return

    def crossover(self, attr_id, attr):
        fir = sec = -1
        for i in range(0, len(self.population)):
            if genetic_algorithm_var.crossover_rates[attr] > random.uniform(0, 1):
                if fir != -1:
                    self.xover(attr_id, attr, fir, i)
                    fir = -1
                else:
                    fir = i
        return

    def xover(self, attr_id, attr,  i, j):
        if attr != "topo":
            si = encode_int(self.population[i].acc_gene[attr_id])
            sj = encode_int(self.population[j].acc_gene[attr_id])
            new_si = si
            new_sj = sj
            for k in range(0, 2):
                tmp = new_si[k]
                new_si[k] = new_sj[k]
                new_sj[k] = tmp
            # sometimes the value is set to 0, so max(1, num)
            self.next_population[i].acc_gene[attr_id] = max(1, decode_int(new_si))
            self.next_population[j].acc_gene[attr_id] = max(1, decode_int(new_sj))
        else:
            pass
        # if attr_type == "pe_num":
        #     self.population[i].acc.pe_num, self.population[j].acc.pe_num = \
        #         self.population[j].acc.pe_num, self.population[i].acc.pe_num
        # elif attr_type == "pe_size":
        #     self.population[i].acc.pe_size, self.population[j].acc.pe_size = \
        #         self.population[j].acc.pe_size, self.population[i].acc.pe_size
        #     pass
        # elif attr_type == "pe_numX":
        #     self.population[i].acc.pe_numX, self.population[j].acc.pe_numX = \
        #         self.population[j].acc.pe_numX, self.population[i].acc.pe_numX
        #     pass
        # elif attr_type == "pe_numY":
        #     self.population[i].acc.pe_numY, self.population[j].acc.pe_numY = \
        #         self.population[j].acc.pe_numY, self.population[i].acc.pe_numY
        #     pass
        # elif attr_type == "global_buffer_size":
        #     self.population[i].acc.global_buf_size, self.population[j].acc.global_buf_size = \
        #         self.population[j].acc.global_buf_size, self.population[i].acc.global_buf_size
        #     pass
        # elif attr_type == "topo":
            # self.population[i].acc.pe_num, self.population[j].acc.pe_num = \
            #     self.population[j].acc.pe_num, self.population[i].acc.pe_num
            # pass
        return

    def mutate(self):
        for p in self.population:
            pass
        return

    def elitist(self):
        best = worst = self.population[0]
        worst_id = 0
        for i in range(0, len(self.population)):
            p = self.population[i]
            if best.fit < p.fit:
                best = p
            if worst.fit > p.fit:
                worst = p
                worst_id = i

        if best.fit > self.best_pop.fit:
            self.best_pop = best
        else:
            self.population[worst_id] = self.best_pop
        return

    def evaluate(self):
        for p in self.population:
            acc = Accelerator(p.acc_gene[1], p.acc_gene[2], p.acc_gene[3], p.acc_gene[4])
            net = Network()
            p.fit = self.cal(acc, net)

    def cal(self, h, net):
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


test_int_encoding()
ga = GeneticAlgorithm()
ga.run()
