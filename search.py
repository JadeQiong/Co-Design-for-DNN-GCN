from accelerator import Accelerator
from network import Network
import global_var
import numpy as np
#电路层级：tile、pe和sram阵列，sram设置为128*128（该数常用），搜索算法目前搜索pe的个数以及pe_x和pe_y
h = Accelerator(10,4,4,10,10)
net = Network([("cov", 20, 20),("cov", 15, 15)])
h.print()
net.print()
subarray=128
# objective function
t_comm = 0
t_comp = 0
num_tile=0
# constraints
G_NUMBER = 10000
area_thres = G_NUMBER
energy_thres = G_NUMBER
accuracy_thres = G_NUMBER
#每个pe包含一个收发路由器，sram阵列以及数模转换器等外围电路，此处求的是一个tile的面积
tile_area = global_var.a_other['pe_buffer']+h.pe_num * (global_var.a_other['router'] + subarray*subarray*global_var.a_cim['sram']+global_var.a_other['others'])
#总面积为tile个数乘以tile面积加Noc路由器面积再加上总缓冲区面积大小，tile数量等于层数
total_area = num_tile*(tile_area+global_var.a_other['router'])+global_var.a_other['global_buffer']
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
    #在这里把mac连接起来,换成PARA的total_mac
    tot_mac = net.layers[i][1] * net.layers[i][2]
    block_mac = tot_mac/h.pe_num
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

# computation time，计算每层pe的最大计算时间
for i in range(0, net.num_Layers):
    max_time = 0
    for j in mapping[i]:
        max_time = max(max_time, mac_count[j] * global_var.t_mac[h.quantization[j]])
        energy += global_var.e_mac[h.quantization[j]] * mac_count[j]
    t_comp += max_time

print("-----------------------------------------")
print("The communication time is {:.2f} ns.".format(t_comm))
print("The computation time is {:.2f} ns.".format(t_comp))
print("Total time is {:.2f} ns.".format(t_comp+t_comm))
print("-----------------------------------------")
print("Area is {:.2f} mm^2.".format(total_area))
print("Energy is {:.2f} pJ.".format(energy))
# print("Accuracy is {:.2f} ns.".format(accuracy))