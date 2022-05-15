from scipy import optimize
import numpy as np
from accelerator import Chip, Chiplet, initiate_topo
from network import Network
import global_var
# x:
# 0-1: chipX,chipY,
# 2-3: tileX, tileY
# 4-5: peX, peY

net = Network(True, [352800, 24893568, 3175200, 6350400])


def cal_chip(chip_arr, mac):
    tile_num = int(chip_arr[0] * chip_arr[1])
    pe_num = int(chip_arr[2] * chip_arr[3])
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
        bit_num = tile_mac_count[p] * data_bit_width
        package_num = int(bit_num / (global_var.package_size * 8))
        tile_max_time = max(tile_max_time,
                            4
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

            bit_num = pe_mac_count[s][j] * data_bit_width
            package_num = int(bit_num / (global_var.package_size * 8))
            pe_max_time = max(pe_max_time,
                            4*
                              global_var.delay_per_hop + global_var.t_trans_per_bit
                              + (global_var.t_package + global_var.t_recv_send) * package_num)
            # energy += (global_var.e_trans_per_bit + global_var.e_cim['dram']) * bit_num
            # 读写 + 传输
            # energy_new += h.tile_numX * h.tile_numY * h.pe_numX * h.pe_numY * (
            #         subarray * subarray * global_var.e_cim['sram'] * bit_num + global_var.e_other['others'] +
            #   global_var.e_other['router']) + global_var.e_trans_per_bit * bit_num
            # pe通信功耗
            # energy_new += global_var.e_trans_per_bit * bit_num
    t_comm += (tile_comm + pe_max_time)

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


def cal(x):
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
    chip_num = int(x[0] * x[1])
    if chip_num == 0:
        return 10
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
            # res = self.cal_chip(chiplet.chips[p], chip_mac_count[i])
            chip = [int(x[2]), int(x[3]), int(x[4]), int(x[5])]
            # print(chip)
            tmp_time, tmp_area, tmp_energy, tmp_error = 20,20,20,20
                # cal_chip(chip, chip_mac_count[p])
            # print("=============== tmp area =========" + str(tmp_area))
            # print("=============== tmp energy =========" + str(tmp_energy))
            total_time = max(total_time, tmp_time)
            total_energy += tmp_energy
            total_area = tmp_area

    total_area *= int(chip_num)
    if total_energy > energy_thres or total_area > area_thres:
        m = (total_energy - energy_thres) * (total_area - area_thres)
    #print()
    # print("=============== total time =========" + str(total_time/pow(10, 6)))
    # print("=============== total area =========" + str(total_area/pow(10, 6)))
    # print("=============== total energy =========" + str(total_energy/pow(10, 12)))
    # print("cost function = " + str(total_time/pow(10, 6)))
    print(total_time/pow(10, 6))
    return (total_time/pow(10, 6))

def f(x):
    return 0.5*(1 - x[0])**2 + (x[1] - 7)**2


# 一阶求导
def fprime(x):
    return np.array((-2*.5*(1 - x[0]), 2*(x[1] - 7)))

def ff(x):
    return x[0]

# res = optimize.fmin(ff, [-1])

# res = optimize.fmin(cal, [2, 2, 2, 2, 2, 2])
# res = optimize.fmin(f, [2, 2])
# print(res)
#
# import matplotlib.pyplot as plt
# def cost_function(x):
#     return (x[0]**2+x[1]-11)**2+(x[0]+x[1]**2-7)**2
#
# n = 50
# x = np.linspace(1, 6, n)
# y = np.linspace(1, 6, n)
# z = np.zeros((n, n))
#
# for i, a in enumerate(x):
#     for j, b in enumerate(y):
#         z[i, j] = cal([a, b, 2, 2, 2, 2])
#
# xx, yy = np.meshgrid(x, y)
# fig, axes = plt.subplots(2, 2, figsize=(12, 8))
#
# centers = [[1, 1, 1, 2, 1, 2], [2, 2, 2, 2, 2, 2], [3, 1, 4, 2, 4, 2], [4, 2, 2, 2, 4, 1]]
# for i, center in enumerate(centers):
#     x_center = np.array(center)
#     step = 0.5
#     #x0 = np.vstack((x_center, x_center+np.diag((step, step))))
#     xtol, ftol = 1e-3, 1e-3
#     #initial_simplex = x0
#     xopt, fopt, iter, funcalls, warnflags, allvecs = \
#         optimize.fmin(cal, x_center, xtol=xtol, ftol=ftol, disp=1, retall=1, full_output=1)
#     print(xopt, fopt)
#
#     ii, jj = i//2, i % 2
#     ax = axes[ii][jj]
#     c = ax.pcolormesh(xx, yy, z.T, cmap='jet')
#     fig.colorbar(c, ax=ax)
#
#     t = np.asarray(allvecs)
#     x_, y_ = t[:, 0], t[:, 1]
#     print("x & y")
#     print("x_  = "+str(x_)+" ,y_ = "+str(y_))
#     ax.plot(x_, y_, 'r', x_[0], y_[0], 'go', x_[-1], y_[-1], 'y+', markersize=60)
#
# fig.show()
#
from scipy.optimize import minimize
from scipy.optimize import Bounds
bounds = Bounds([1,1,1,1,1,1],[5,5,10,10,10,10])
x0 = [3,6,8,2,2,2]
res = minimize(cal, x0, options={'ftol': 1e-9, 'disp': True},
               bounds=bounds)
print(res.x)
