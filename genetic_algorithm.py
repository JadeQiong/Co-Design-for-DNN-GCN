from accelerator import Accelerator
from network import Network
import ga_configs
import global_var
import random
import copy
import numpy as np

# 遗传算法搜索对象：pe数量，pe阵列的行和列，拓扑结构


class Population:
    def __init__(self, acc=Accelerator(), net=Network(), fit=0, r_fit=0, c_fit=0):
        self.acc_gene = [acc.pe_numX, acc.pe_numY, acc.pe_size, acc.global_buf_size, acc.topo]
        self.net_gene = [net.num_Layers, net.layers, net.layer_connection]
        self.fit = fit  # 适应度
        self.r_fit = r_fit   # 轮盘适应度
        self.c_fit = c_fit   # 累计适应度


# 对pe数量编码
def encode_int(num):
    s = []
    while int(num) != 0:
        s.append(num % 2)
        num = int(num/2)
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

# decode_int([0,0,0,1,0])


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
    id = 0
    for i in range(0, max_len):
        for j in range(i+1, max_len):
            topo_padding[j][i] = topo_padding[i][j] = s[id]
            id += 1
    return topo_padding[0:x, 0:x]


def test_topo_encoding():
    for i in range(200):
        m = random.randint(1, 10)
        a = np.zeros((m, m))
        for ii in range(0, m):
            for jj in range(ii+1, m):
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


class GeneticAlgorithm:

    def __init__(self, pop_num=10, iter_num=10, gen_num=10, pf=1):
        self.pop_num = pop_num   # 种群数量
        self.iter_num = iter_num  # 迭代次数
        self.gen_num = gen_num   # 迭代一次的时候有多少代种群产生
        self.p_factor = pf      # punish factor
        self.population = [Population() for i in range(self.pop_num)]    # 当前种群
        self.next_population = [Population() for i in range(self.pop_num)]   # 下一代种群
        self.best_pop = self.population[0]

    def initiate(self):
        for p in self.population:
            # hardware init
            for j in range(0, len(ga_configs.acc_gene_type)):
                gene = ga_configs.acc_gene_type[j]
                if gene == "pe_numX":
                    p.acc_gene[j] = random.randint(1, 5)
                elif gene == "pe_numY":
                    p.acc_gene[j] = random.randint(1, 5)
                elif gene == "pe_size":
                    p.acc_gene[j] = random.randint(1, 5)
                elif gene == "global_buffer_size":
                    p.acc_gene[j] = random.randint(1, 10)
                elif gene == "topo":
                    # num = x*y
                    pe_num = p.acc_gene[0] * p.acc_gene[1]
                    p.acc_gene[j] = random_topo(pe_num)
            # software init

            # GG
        return

    # 保存当前最优解
    def keep_the_best(self):
        for p in self.population:
            if self.best_pop.fit < p.fit:
                self.best_pop = p

    # 适者生存
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
            probability = random.random()
            if probability < self.population[0].c_fit:
                self.next_population[i] = copy.deepcopy(self.population[i])
            else:
                for j in range(0, len(self.population)-1):
                    if self.population[j].c_fit <= probability < self.population[j + 1].c_fit:
                        self.next_population[i] = copy.deepcopy(self.population[j+1])

        # update population
        self.population = copy.deepcopy(self.next_population)
        return

    # 选一对个体杂交
    def crossover(self, attr_id):
        fir = -1
        attr = ga_configs.acc_gene_type[attr_id]
        for i in range(0, len(self.population)):
            p = random.uniform(0, 1)
            if ga_configs.crossover_rates[attr] > p:
                if fir != -1:
                    self.xover(attr_id, attr, fir, i)
                    fir = -1
                else:
                    fir = i
        return

    # 双亲杂交产生新的个体
    def xover(self, attr_id, attr,  i, j):
        if attr != "topo":
            si = encode_int(self.population[i].acc_gene[attr_id])
            sj = encode_int(self.population[j].acc_gene[attr_id])
        else:
            si = encode_topo(self.population[i].acc_gene[attr_id])
            sj = encode_topo(self.population[i].acc_gene[attr_id])

        new_si = copy.deepcopy(si)
        new_sj = copy.deepcopy(sj)

        for k in range(0, 45 if attr == "topo" else 2):
            tmp = new_si[k]
            new_si[k] = new_sj[k]
            new_sj[k] = tmp

        if attr != "topo":
            # 先试试交换
            self.next_population[i].acc_gene[attr_id] = self.population[j].acc_gene[attr_id]
            self.next_population[j].acc_gene[attr_id] = self.population[i].acc_gene[attr_id]
            # self.next_population[i].acc_gene[attr_id] = max(1, decode_int(new_si))
            # self.next_population[j].acc_gene[attr_id] = max(1, decode_int(new_sj))
        else:
            pe_numi = self.next_population[i].acc_gene[0] * self.next_population[i].acc_gene[1]
            pe_numj = self.next_population[j].acc_gene[0] * self.next_population[j].acc_gene[1]
            self.next_population[i].acc_gene[attr_id] = decode_topo(new_si, pe_numi)
            self.next_population[j].acc_gene[attr_id] = decode_topo(new_sj, pe_numj)

        print(" -------parents------- ")
        print(self.population[i].acc_gene[0:3])
        print(self.population[j].acc_gene[0:3])
        print(" ------children----- ")
        print(self.next_population[i].acc_gene[0:3])
        print(self.next_population[j].acc_gene[0:3])
        print(" --------------------------------- ")

        return

    # 随机变异
    def mutate(self):
        for p in self.population:
            probability = []
            for i in range(0, len(p.acc_gene)):
                probability.append(random.random())
            for i in range(0, len(p.acc_gene)):
                attr = ga_configs.acc_gene_type[i]
                if probability[i] < ga_configs.mutate_rates[attr]:
                    if attr == "pe_size":
                        p.acc_gene[i] = random.randint(1, 5)
                    elif attr == "global_pe_buffer":
                        p.acc_gene[i] = random.randint(1, 10)
                    elif attr == "topo":
                        pe_num = p.acc_gene[0] * p.acc_gene[1]
                        for ii in range(0, pe_num):
                            for jj in range(pe_num):
                                pp = random.random()
                                if pp<ga_configs.mutate_rates[attr]:
                                   p.acc_gene[i][ii][jj] = 1 - p.acc_gene[i][ii][jj]
                        #p.acc_gene[i] = random_topo(pe_num)

            if p.acc_gene[0] * p.acc_gene[1] != len(p.acc_gene[4]):
                print(p.acc_gene[0])
                print(p.acc_gene[1])
                print(len(p.acc_gene[4]))
                print("G了，如果看到这条需要提醒debug")
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
            if p.acc_gene[0] * p.acc_gene[1] != len(p.acc_gene[4]):
                print("???????????????")
                continue
            acc = Accelerator(p.acc_gene[0], p.acc_gene[1], p.acc_gene[2], p.acc_gene[3], p.acc_gene[4])
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
        area = h.pe_num * (global_var.a_other['router'] + global_var.a_cim['sram'])
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

    def run(self):
        self.initiate()
        self.evaluate()
        self.keep_the_best()
        for attr_id in range(0, len(ga_configs.acc_gene_type)):
            self.crossover(attr_id)

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
                print(self.best_pop.acc_gene[0:4])
                print(self.best_pop.net_gene[0:4])

        return


# test_int_encoding()
# if test_topo_encoding() == False:
#     print("G")
ga = GeneticAlgorithm()
ga.run()
