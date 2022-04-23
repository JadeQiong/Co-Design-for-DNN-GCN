from accelerator import Accelerator
from network import Network
import ga_configs
import global_var
import random
import copy
import numpy as np


# 遗传算法搜索对象：pe数量，pe阵列的行和列，拓扑结构


class Population:
    def __init__(self, acc=Accelerator(), net=Network([[20, 20, 30], [20, 30, 30]]), fit=0, r_fit=0, c_fit=0):
        self.acc_gene = [acc.pe_numX, acc.pe_numY, acc.tile_numX, acc.tile_numY,
                         acc.pe_size, acc.global_buf_size, acc.pe_topo, acc.tile_topo]
        self.net = net
        # useless, but we leave it here temporarily
        self.net_gene = [net.macs]
        self.fit = fit  # 适应度
        self.r_fit = r_fit  # 轮盘适应度
        self.c_fit = c_fit  # 累计适应度

    def pe_num(self):
        return self.acc_gene[0] * self.acc_gene[1]

    def tile_num(self):
        return self.acc_gene[2] * self.acc_gene[3]


# 对pe/tile数量编码
def encode_int(num):
    s = []
    while int(num) != 0:
        s.append(num % 2)
        num = int(num / 2)
    dif = ga_configs.int_code_len - len(s)
    for i in range(0, dif):
        s.append(0)
    # 把这个reverse先注释掉了
    # s.reverse()
    return s


def decode_int(s):
    num = 0
    # s.reverse()
    for i in range(0, len(s)):
        if s[i] == 1:
            num += pow(2, i)
    return num


def encode_float(num):
    return


def decode_float(s):
    return


def random_topo(m):
    topo = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            topo[i][j] = random.randint(0, 1)
    return topo


def encode_topo(topo):
    s = []
    x = len(topo)
    max_len = ga_configs.topo_code_len
    topo_padding = np.zeros((max_len, max_len))
    for i in range(0, max_len):
        for j in range(0, max_len):
            if 0 <= i < x and 0 <= j < x:
                topo_padding[i][j] = topo[i][j]
    # 展开
    for i in range(0, max_len):
        for j in range(i + 1, max_len):
            s.append(topo_padding[i][j])
    return s


def decode_topo(s, x):
    max_len = ga_configs.topo_code_len
    topo_padding = np.zeros((max_len, max_len))
    topo_id = 0
    for i in range(0, max_len):
        for j in range(i + 1, max_len):
            topo_padding[j][i] = topo_padding[i][j] = s[topo_id]
            topo_id += 1
    return topo_padding[0:x, 0:x]


def test_topo_encoding():
    for i in range(200):
        m = random.randint(1, 10)
        a = np.zeros((m, m))
        for ii in range(0, m):
            for jj in range(ii + 1, m):
                p = random.random()
                if ii != jj:
                    a[ii][jj] = a[jj][ii] = 1 if p < 0.5 else 0
        b = decode_topo(encode_topo(a), m)
        if len(b) != len(a):
            return False
        for i in range(len(b)):
            for j in range(len(a)):
                if b[i][j] != a[i][j]:
                    return False
        # if decode_topo(encode_topo(a), m).all() != a.all():
        #     print(a)
        #     print(encode_topo(a))
        #     print(decode_topo(encode_topo(a), m))
        #     print("--------------")
        #     print("G")
    return True


def test_int_encoding():
    for i in range(100):
        if decode_int(encode_int(i)) != i:
            print(i)
            print(encode_int(i))
            print(decode_int(encode_int(i)))
            print("test failed")


def print_res(res):
    print("pe_num = " + str(res[0]) + " * " + str(res[1]) + " = " + str(res[0] * res[1]))
    print("pe_size = " + str(res[2]))
    print("pe_global_buffer = " + str(res[3]))


class GeneticAlgorithm:

    def __init__(self, pop_num=10, iter_num=10, gen_num=10, pf=1):
        self.pop_num = pop_num  # 种群数量
        self.iter_num = iter_num  # 迭代次数
        self.gen_num = gen_num  # 迭代一次的时候有多少代个体产生
        self.p_factor = pf  # punish factor
        self.population = [Population() for i in range(self.pop_num)]  # 当前种群
        self.next_population = [Population() for i in range(self.pop_num)]  # 下一代种群
        self.best_pop = self.population[0]

    def set_network(self, net):
        for p in self.population:
            p.net = net

    def initiate(self):
        for p in self.population:
            # hardware init
            for j in range(len(ga_configs.acc_gene_type)):
                gene = ga_configs.acc_gene_type[j]
                if gene == "pe_numX":
                    p.acc_gene[j] = random.randint(1, 5)
                elif gene == "pe_numY":
                    p.acc_gene[j] = random.randint(1, 5)
                elif gene == "tile_numX":
                    p.acc_gene[j] = random.randint(1, 5)
                elif gene == "tile_numY":
                    p.acc_gene[j] = random.randint(1, 5)
                elif gene == "pe_size":
                    p.acc_gene[j] = random.randint(1, 5)
                elif gene == "global_buffer_size":
                    p.acc_gene[j] = random.randint(1, 10)
                elif gene == "pe_topo":
                    p.acc_gene[j] = random_topo(p.pe_num())
                elif gene == "tile_topo":
                    p.acc_gene[j] = random_topo(p.tile_num())
                print("gene " + gene + str("id = ") + str(p.acc_gene[j]))
            # software init
        return

    # 保存当前最优解
    def keep_the_best(self):
        for p in self.population:
            if self.best_pop.fit < p.fit:
                self.best_pop = p

    # 适者生存
    def select(self):
        fit_sum = 2
        # 轮盘赌
        for p in self.population:
            fit_sum += p.fit
        for p in self.population:
            p.r_fit = p.fit / fit_sum

        for i in range(0, len(self.population)):
            if i == 0:
                self.population[i].c_fit = self.population[i].fit
            else:
                self.population[i].c_fit = self.population[i - 1].c_fit + self.population[i].fit

        # generate random probability
        for i in range(0, len(self.population)):
            probability = random.random()
            if probability < self.population[0].c_fit:
                self.next_population[i] = copy.deepcopy(self.population[i])
            else:
                for j in range(0, len(self.population) - 1):
                    if self.population[j].c_fit <= probability < self.population[j + 1].c_fit:
                        self.next_population[i] = copy.deepcopy(self.population[j + 1])

        # update population
        self.population = copy.deepcopy(self.next_population)
        return

    # 近似：分段线性
    # 多项式拟合
    #
    # 选一对个体杂交
    def crossover(self, attr_id):
        fir = -1
        attr = ga_configs.acc_gene_type[attr_id]
        for i in range(0, len(self.population)):
            p = random.uniform(0, 1)
            if ga_configs.crossover_rates[attr] > p:
                if fir != -1:
                    print("attr_id: " + str(attr_id) + ", " + str(attr))
                    self.xover(attr_id, attr, fir, i)
                    fir = -1
                else:
                    fir = i
        return

    # 双亲杂交产生新的个体
    def xover(self, attr_id, attr, i, j):
        if (attr != "tile_topo") and (attr != "pe_topo"):
            si = encode_int(self.population[i].acc_gene[attr_id])
            sj = encode_int(self.population[j].acc_gene[attr_id])
        else:
            si = encode_topo(self.population[i].acc_gene[attr_id])
            sj = encode_topo(self.population[j].acc_gene[attr_id])

        new_si = copy.deepcopy(si)
        new_sj = copy.deepcopy(sj)

        for k in range(0, 45 if (attr == "tile_topo") and (attr == "pe_topo") else 2):
            tmp = new_si[k]
            new_si[k] = new_sj[k]
            new_sj[k] = tmp

        if (attr != "tile_topo") and (attr != "pe_topo"):
            # 先试试交换
            self.next_population[i].acc_gene[attr_id] = self.population[j].acc_gene[attr_id]
            self.next_population[j].acc_gene[attr_id] = self.population[i].acc_gene[attr_id]
            # self.next_population[i].acc_gene[attr_id] = max(1, decode_int(new_si))
            # self.next_population[j].acc_gene[attr_id] = max(1, decode_int(new_sj))
        elif attr == "pe_topo":
            self.next_population[i].acc_gene[attr_id] = decode_topo(new_si, self.next_population[i].pe_num())
            self.next_population[j].acc_gene[attr_id] = decode_topo(new_sj, self.next_population[j].pe_num())
        else:
            self.next_population[i].acc_gene[attr_id] = decode_topo(new_si, self.next_population[i].tile_num())
            self.next_population[j].acc_gene[attr_id] = decode_topo(new_sj, self.next_population[j].tile_num())
        print(" -------parents------- ")
        # print(self.population[i].acc_gene[6])
        # print(self.population[j].acc_gene[6])
        print(" ------children----- ")
        print(self.next_population[i].acc_gene[6])
        print(self.next_population[j].acc_gene[6])
        print(" --------------------------------- ")

        return

    # 随机变异
    def mutate(self):
        for p in self.population:
            probability = []
            for i in range(len(p.acc_gene)):
                probability.append(random.random())
            for i in range(len(p.acc_gene)):
                attr = ga_configs.acc_gene_type[i]
                if probability[i] < ga_configs.mutate_rates[attr]:
                    if attr == "pe_size":
                        p.acc_gene[i] = random.randint(1, 5)
                    elif attr == "global_pe_buffer":
                        p.acc_gene[i] = random.randint(1, 10)
                    elif attr == "pe_topo" or attr == "tile_topo":
                        pass
                        # num = p.pe_num() if attr == "pe_topo" else p.tile_num()
                        # print("size = " + str(len(p.acc_gene[i])))
                        # print("num = " + str(num))
                        # for q in range(num):
                        #     for t in range(num):
                        #         pp = random.random()
                        #         if pp < ga_configs.mutate_rates[attr]:
                        #            p.acc_gene[i][q][t] = 1 - p.acc_gene[i][q][t]

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
            print("pe_numX " + str(p.acc_gene[0]))
            acc = Accelerator(True, p.acc_gene)
            print(acc.pe_numX)
            net = p.net
            p.fit = self.cal(acc, net)

    def cal(self, h, net):
        # objective function
        t_comm = 0
        t_comp = 0

        # 乘法因子
        m = 0

        # constraints
        g_number = 10000
        area_thres = g_number
        energy_thres = g_number
        accuracy_thres = g_number
        subarray = 128
        tile_area = global_var.a_other['pe_buffer'] + h.pe_numX * h.pe_numY * \
                    (global_var.a_other['router'] + subarray * subarray * global_var.a_cim['sram'] + global_var.a_other[
                        'others'])
        # 总面积为tile个数乘以tile面积加Noc路由器面积再加上总缓冲区面积大小，tile数量等于层数
        area = h.tile_numX * h.tile_numY * (tile_area + global_var.a_other['router']) + global_var.a_other[
            'global_buffer']
        energy = 0
        accuracy = 0

        # communication time
        mapping = [[] for i in range(net.num_Layers)]
        accumulate_pe_id = [0 for i in range(net.num_Layers)]
        mac_count = [0 for i in range(h.pe_numX * h.pe_numY)]
        data_bit_width = 64

        # print("num of layers = "+str(net.num_Layers))
        for i in range(0, net.num_Layers):
            # ！这里不会算
            # 方案1：平均分配给每个PE
            # print("mac = " + str(net.macs[i]))
            tot_mac = net.macs[i]
            tile_block_mac = tot_mac / (h.tile_numX * h.tile_numY)
            pe_block_mac = tile_block_mac / (h.pe_numX * h.pe_numY)
            # for each layer, we assign an accumulating PE(id = 0) for summing up partial sums
            accumulate_pe_id[i] = 0
            for j in range(h.pe_numX * h.pe_numY):
                # mapping[i]: the i th layer is mapped to PE[0th ,1th, 2th, ... ]
                mac_count[j] = pe_block_mac
                mapping[i].append(j)

        for i in range(net.num_Layers):
            max_hop = 0
            for j in mapping[i]:
                max_hop = max(max_hop, h.pe_topo_dis(accumulate_pe_id[i], j))
                bit_num = mac_count[j] * data_bit_width
                energy += global_var.e_trans * bit_num
            t_comm += max_hop * (global_var.t_trans + global_var.t_package + global_var.t_package)

        # computation time
        for i in range(0, len(net.macs)):
            max_time = 0
            for j in mapping[i]:
                max_time = max(max_time, mac_count[j] * 6)
                # global_var.t_mac[h.quantization[j]])
                # energy += global_var.e_mac[h.quantization[j]] * mac_count[j]
                energy += 6 * mac_count[j]
            t_comp += max_time

        if energy > energy_thres or area > area_thres or accuracy < accuracy_thres:
            m = (energy - energy_thres) * (area - area_thres) * (accuracy - accuracy_thres)
        print("time = " + str(t_comm + t_comp))
        print("area = " + str(area))
        print("energy = " + str(energy))
        print(" -----------------------------------")
        # TODO: return fitness function
        return 1 / max((t_comm + t_comp), 1) * self.p_factor * m

    def run(self):
        self.initiate()
        self.evaluate()
        self.keep_the_best()
        for attr_id in range(0, len(ga_configs.acc_gene_type)):
            self.crossover(attr_id)
            print(attr_id)

        gen = 0
        for cur_iter in range(0, self.iter_num):
            while gen < self.gen_num:
                gen += 1
                self.select()
                # 对每个基因型进行交叉杂交
                for attr_id in range(1, len(ga_configs.acc_gene_type)):
                    self.crossover(attr_id)
                self.mutate()
                self.evaluate()
                self.keep_the_best()
                self.elitist()
                print("----------------best in each iteration " + str(gen) + "----------------")
                print_res(self.best_pop.acc_gene)
                print(self.best_pop.net_gene[0:4])
        return


# test_int_encoding()
# if test_topo_encoding() == False:
#     print("G")
ga = GeneticAlgorithm()
ga.run()
