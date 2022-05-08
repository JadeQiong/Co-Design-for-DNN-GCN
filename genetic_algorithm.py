# -*- coding: UTF-8 -*-
from accelerator import Chip, Chiplet, initiate_topo
from network import Network
import ga_configs
import global_var
import random
import copy
import numpy as np


# 遗传算法搜索对象：pe数量，pe阵列的行和列，拓扑结构


class Population:
    def __init__(self, acc=Chip(), chiplet=Chiplet(), net=Network(True, [352800, 24893568, 3175200, 6350400]), fit=0,
                 r_fit=0, c_fit=0):
        self.acc_gene = [acc.pe_numX, acc.pe_numY, acc.tile_numX, acc.tile_numY,
                         acc.pe_size, acc.global_buf_size, acc.pe_topo, acc.tile_topo]
        self.chiplet_gene = [chiplet.chipX, chiplet.chipY, chiplet.chiplet_topo]
        self.net = net
        # useless, but we leave it here temporarily
        self.net_gene = [net.macs]
        self.fit = fit  # 适应度
        self.r_fit = r_fit  # 轮盘适应度
        self.c_fit = c_fit  # 累计适应度
        self.area = -1
        self.energy = -1
        self.acc = -1
        self.time = -1

    def pe_num(self):
        return self.acc_gene[0] * self.acc_gene[1]

    def tile_num(self):
        return self.acc_gene[2] * self.acc_gene[3]

    def chip_num(self):
        return self.chiplet_gene[0] * self.chiplet_gene[1]


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

    def __init__(self, pop_num=1, iter_num=1, gen_num=5, pf=1):
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
            for j in range(len(ga_configs.chiplet_gene_type)):
                gene = ga_configs.chiplet_gene_type[j]
                if gene == "chipX" or gene == "chipY":
                    p.chiplet_gene[j] = random.randint(1, 4)
                    p.chiplet_gene[j] = random.randint(1, 4)
                elif gene == "chiplet_topo":
                    p.chiplet_gene[j] = random_topo(p.chip_num())

            for j in range(len(ga_configs.acc_gene_type)):
                gene = ga_configs.acc_gene_type[j]
                if gene == "pe_numX":
                    p.acc_gene[j] = random.randint(3, 5)
                elif gene == "pe_numY":
                    p.acc_gene[j] = random.randint(3, 5)
                elif gene == "tile_numX":
                    p.acc_gene[j] = random.randint(1, 3)
                elif gene == "tile_numY":
                    p.acc_gene[j] = random.randint(1, 3)
                elif gene == "pe_size":
                    p.acc_gene[j] = random.randint(1, 5)
                elif gene == "global_buffer_size":
                    p.acc_gene[j] = random.randint(1, 10)
                elif gene == "pe_topo":
                    p.acc_gene[j] = random_topo(p.pe_num())
                elif gene == "tile_topo":
                    p.acc_gene[j] = random_topo(p.tile_num())
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

    def crossover(self):
        # 选一对个体杂交
        self.crossover_acc(gene_type="acc")
        self.crossover_acc(gene_type="chiplet")

    def crossover_acc(self, gene_type):
        gene_len = len(ga_configs.acc_gene_type) if gene_type == "acc" else len(ga_configs.chiplet_gene_type)
        for attr_id in range(gene_len):
            fir = -1
            attr = ga_configs.acc_gene_type[attr_id] if gene_type == "acc" else ga_configs.chiplet_gene_type[attr_id]
            for i in range(len(self.population)):
                p = random.uniform(0, 1)
                if ga_configs.crossover_rates[attr] > p:
                    if fir != -1:
                        self.xover(attr_id, attr, fir, i, gene_type)
                        fir = -1
                    else:
                        fir = i
        return

    # 双亲杂交产生新的个体
    def xover(self, attr_id, attr, i, j, gene_type):
        if gene_type == "acc":
            if (attr != "tile_topo") and (attr != "pe_topo"):
                # 如果是int类型
                si = encode_int(self.population[i].acc_gene[attr_id])
                sj = encode_int(self.population[j].acc_gene[attr_id])
            else:
                # 如果是topo类型
                si = encode_topo(self.population[i].acc_gene[attr_id])
                sj = encode_topo(self.population[j].acc_gene[attr_id])

            new_si = copy.deepcopy(si)
            new_sj = copy.deepcopy(sj)

            for k in range(0, 45 if (attr == "tile_topo" or attr == "pe_topo") else 2):
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
            elif attr == "tile_topo":
                self.next_population[i].acc_gene[attr_id] = decode_topo(new_si, self.next_population[i].tile_num())
                self.next_population[j].acc_gene[attr_id] = decode_topo(new_sj, self.next_population[j].tile_num())

            si = encode_topo(self.population[i].acc_gene[6])
            self.next_population[i].acc_gene[6] = decode_topo(si, self.next_population[i].pe_num())
            sj = encode_topo(self.population[j].acc_gene[6])
            self.next_population[j].acc_gene[6] = decode_topo(sj, self.next_population[j].pe_num())
            ti = encode_topo(self.population[i].acc_gene[7])
            self.next_population[i].acc_gene[7] = decode_topo(ti, self.next_population[i].tile_num())
            tj = encode_topo(self.population[j].acc_gene[7])
            self.next_population[j].acc_gene[7] = decode_topo(tj, self.next_population[j].tile_num())
            # print(" -------parents------- ")
            # # print(self.population[i].acc_gene[6])
            # # print(self.population[j].acc_gene[6])
            # print(" ------children----- ")
            # print(self.next_population[i].acc_gene[6])
            # print(self.next_population[j].acc_gene[6])
            # print(" --------------------------------- ")
        elif gene_type == "chiplet":
            if attr == "chipX" or attr == "chipY":
                # 如果是int类型
                si = encode_int(self.population[i].chiplet_gene[attr_id])
                sj = encode_int(self.population[j].chiplet_gene[attr_id])
            elif attr == "chiplet_topo":
                # 如果是topo类型
                si = encode_topo(self.population[i].chiplet_gene[attr_id])
                sj = encode_topo(self.population[j].chiplet_gene[attr_id])

            new_si = copy.deepcopy(si)
            new_sj = copy.deepcopy(sj)

            for k in range(0, 45 if attr == "chiplet_topo" else 2):
                tmp = new_si[k]
                new_si[k] = new_sj[k]
                new_sj[k] = tmp
            if attr != "chiplet_topo":
                self.next_population[i].chiplet_gene[attr_id], self.next_population[j].chiplet_gene[attr_id] = \
                    self.next_population[j].chiplet_gene[attr_id], self.next_population[i].chiplet_gene[attr_id]
            else:
                self.next_population[i].chiplet_gene[attr_id] = decode_topo(new_si, self.next_population[i].chip_num())
                self.next_population[j].chiplet_gene[attr_id] = decode_topo(new_si, self.next_population[i].chip_num())

            si = encode_topo(self.population[i].chiplet_gene[2])
            self.next_population[i].chiplet_gene[2] = decode_topo(si, self.next_population[i].chip_num())
            sj = encode_topo(self.population[j].chiplet_gene[2])
            self.next_population[j].chiplet_gene[2] = decode_topo(sj, self.next_population[j].chip_num())

        return

    # 随机变异
    def mutate(self):
        print("muate ????????????????????????")
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
                        num = p.pe_num() if attr == "pe_topo" else p.tile_num()
                        # print("size = " + str(len(p.acc_gene[i])))
                        # print("num = " + str(num))
                        for q in range(num):
                            for t in range(num):
                                pp = random.random()
                                if pp < ga_configs.mutate_rates[attr]:
                                    p.acc_gene[i][q][t] = 1 - p.acc_gene[i][q][t]
                        # pass

            for i in range(len(p.chiplet_gene)):
                pro = random.random()
                attr = ga_configs.chiplet_gene_type[i]
                if pro < ga_configs.mutate_rates[attr]:
                    if attr == "chipX" or attr == "chipY":
                        p.chiplet_gene[i] = random.randint(1, 4)
                        si = encode_topo(p.chiplet_gene[2])
                        p.chiplet_gene[2] = decode_topo(si, p.chip_num())
                    elif attr == "chiplet_topo":
                        num = p.chip_num()
                        for q in range(num):
                            for t in range(num):
                                pp = random.random()
                                if pp < ga_configs.mutate_rates[attr]:
                                    p.chiplet_gene[i][q][t] = 1 - p.acc_gene[i][q][t]
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
            chiplet_topo = np.zeros((16, 16))
            initiate_topo(chiplet_topo, 4, 4)
            chips = [Chip(uselist=True, acc_gene=p.acc_gene) for i in range(16)]
            chiplet = Chiplet(useDefault=False, chips=chips, chiplet_topo=chiplet_topo)
            net = p.net
            res = self.cal_chiplet(chiplet, net)
            p.fit, p.time, p.area, p.energy, p.acc = res[0], res[1], res[2], res[3], res[4]

    def cal_chiplet(self, chiplet, net):
        # 乘法因子
        m = 0
        # constraints
        g_number = 100000
        area_thres = 40 * pow(10, 6)
        energy_thres = pow(10, 13)
        error_thres = g_number

        total_time = 0
        total_energy = 0
        total_area = 0
        total_error = 0

        chip_mapping = [[] for i in range(net.num_Layers)]
        chip_num = chiplet.chipX * chiplet.chipY
        chip_mac_count = [0 for i in range(chip_num)]

        for i in range(0, net.num_Layers):
            # ！这里不会算
            chip_block_mac = net.macs[i] / chip_num
            for p in range(chip_num):
                # tile_mapping[i]: the i th layer is mapped to tile[0th ,1th, 2th, ... ]
                chip_mac_count[p] += chip_block_mac
                # 方案1：平均分配给每个chip
                chip_mapping[i].append(p)

        for i in range(len(chip_mapping)):
            for p in chip_mapping[i]:
                res = self.cal_chip(chiplet.chips[p], chip_mac_count[i])
                tmp_time, tmp_area, tmp_energy, tmp_error = self.cal_chip(chiplet.chips[i], chip_mac_count[i])
                # print("=============== tmp time =========" + str(tmp_time))
                # print("=============== tmp area =========" + str(tmp_area))
                # print("=============== tmp energy =========" + str(tmp_energy))
                total_time += tmp_time
                total_energy += tmp_energy
                total_area = tmp_area

        total_area *= chip_num

        if total_energy > energy_thres or total_area > area_thres:
            m = (total_energy - energy_thres) * (total_area - area_thres)

        print("=============== total time =========" + str(total_time/pow(10, 6)))
        print("=============== total area =========" + str(total_area/pow(10, 6)))
        print("=============== total energy =========" + str(total_energy/pow(10, 12)))

        print(1 / (max(1, total_time) / pow(10, 6)))
        # 1 / (max(1, total_time) / pow(10, 6) + self.p_factor * m)
        return 1 / (max(1, total_time) / pow(10, 6)), total_time, total_area, total_energy, total_error

    def cal_chip(self, chip, mac):
        tile_num = chip.tile_numX * chip.tile_numY
        pe_num = chip.pe_numX * chip.pe_numY
        t_comm = 0
        t_comp = 0
        subarray = 128
        tile_area = global_var.a_other['pe_buffer'] + pe_num * (
                global_var.a_other['router'] + subarray * subarray * global_var.a_cim['sram'] + global_var.a_other[
            'periphery'])

        # 总面积为tile个数乘以tile面积加Noc路由器面积再加上总缓冲区面积大小，tile数量等于层数
        total_area = tile_num * (tile_area + global_var.a_other['router']) + global_var.a_other['global_buffer'] + \
                     global_var.a_other['periphery']
        # print("tile area = " + str(total_area))
        energy = 0
        energy_new = 0
        accuracy = 0

        # communication time
        pe_mapping = [[] for j in range(tile_num)]
        # for each layer with macs = mac, we assign tile_id = 0 for accumulating
        accumulate_tile_id = 0
        accumulate_pe_id = [0 for q in range(tile_num)]
        tile_mac_count = [mac / tile_num for i in range(tile_num)]
        pe_mac_count = [[mac / (tile_num * pe_num) for i in range(pe_num)]
                        for j in range(tile_num)]

        tile_max_time = 0
        pe_max_time = 0
        data_bit_width = 16
        for s in range(tile_num):
            for j in range(pe_num):
                pe_mapping[s].append(j)

        # tile层通信
        for p in range(tile_num):
            if accumulate_tile_id >= len(chip.tile_topo) or p >= len(chip.tile_topo):
                print("tile topo = " + str(chip.tile_topo.shape))
                print("tile dis between " + str(accumulate_tile_id) + " " + str(p))
                continue
            bit_num = tile_mac_count[p] * data_bit_width
            package_num = int(bit_num / (global_var.package_size * 8))
            tile_max_time = max(tile_max_time,
                                chip.tile_topo_dis(accumulate_tile_id, p)
                                * global_var.delay_per_hop + global_var.t_trans_per_bit
                                + (global_var.t_package + global_var.t_recv_send) * package_num)
            energy += global_var.e_trans_per_bit * bit_num
            # 传输
            # tile通信功耗
            energy_new += global_var.e_trans_per_bit * bit_num

        tile_comm = tile_max_time
        # tile层其他功耗
        energy_new += tile_num * global_var.e_other['router'] + global_var.e_other['periphery']
        # pe层通信
        for s in range(0, tile_num):
            for j in pe_mapping[s]:
                if accumulate_pe_id[s] >= len(chip.pe_topo) or j >= len(chip.pe_topo):
                    print("pe topo = " + str(chip.pe_topo.shape))
                    print("dis between " + str(accumulate_pe_id[s]) + " " + str(j))
                    continue

                bit_num = pe_mac_count[s][j] * data_bit_width
                package_num = int(bit_num / (global_var.package_size * 8))
                pe_max_time = max(pe_max_time,
                                  chip.pe_topo_dis(accumulate_pe_id[s], j) *
                                  global_var.delay_per_hop + global_var.t_trans_per_bit
                                  + (global_var.t_package + global_var.t_recv_send) * package_num)
                # energy += (global_var.e_trans_per_bit + global_var.e_cim['dram']) * bit_num
                # 读写 + 传输
                # energy_new += h.tile_numX * h.tile_numY * h.pe_numX * h.pe_numY * (
                #         subarray * subarray * global_var.e_cim['sram'] * bit_num + global_var.e_other['others'] +
                #   global_var.e_other['router']) + global_var.e_trans_per_bit * bit_num
                # pe通信功耗
                # energy_new += global_var.e_trans_per_bit * bit_num
            # pe_comm += pe_max_time[s] *  + global_var.t_package + global_var.t_package
            pe_comm = pe_max_time
        t_comm += (tile_comm + pe_comm)

        # computation time
        max_comm_time = 0
        for s in range(0, tile_num):
            for j in pe_mapping[s]:
                # max_time = max(max_time, pe_mac_count[s][j] * global_var.t_mac[h.quantization[j]])
                max_comm_time = max(max_comm_time, pe_mac_count[s][j] * global_var.t_mac['16b'])
                # energy += global_var.e_mac[h.quantization[j]] * pe_mac_count[s][j]
                energy += global_var.e_mac['8b'] * pe_mac_count[s][j]
        t_comp += max_comm_time
        # 计算功耗
        energy_new += tile_num * pe_num * global_var.e_cim['sram'] + tile_num * global_var.e_other['periphery']

        energy_new *= (t_comm + t_comp)
        return t_comp + t_comm, total_area, energy_new, accuracy

    def run(self):
        self.initiate()
        print("initiated")
        self.evaluate()
        print("evaluated")
        self.keep_the_best()
        print("keep the best")
        self.crossover()
        gen = 0
        for cur_iter in range(0, self.iter_num):
            print("iter = " + str(cur_iter))
            while gen < self.gen_num:
                gen += 1
                self.select()
                self.mutate()
                self.crossover()
                self.evaluate()
                self.keep_the_best()
                print(" ============ " + str(self.best_pop.fit) + "==============")
                self.elitist()
                # print("----------------best in each iteration " + str(gen) + "----------------")
                # print_res(self.best_pop.acc_gene)
                # print(self.best_pop.net_gene[0:4])
        return


# TODO: 遗传算法的优化
# 1. 放弃轮盘赌策略，使用fitness函数排序，排在前面的个体复制2份，中间一份，后面的放弃
# 2. 择优交叉变异
# 3. 防止优良基因被变异掉

# test_int_encoding()
# if test_topo_encoding() == False:
#     print("G")


ga = GeneticAlgorithm()
ga.run()
best = ga.best_pop
print("~~ best gene ~~")
print(ga.best_pop.acc_gene)
print("~~  ------------------- ~~")
print("best: time = " + str(best.time) + " ns = " + str(best.time / pow(10, 6)) + " ms")
print("area = " + str(best.area) + " um^2 = " + str(best.area / pow(10, 6)) + " mm^2")
print("energy = " + str(best.energy / pow(10, 12)) + " * 10^12 pJ")
