# HARDWARE
# time: ns, energy: pJ/bit, area: um^2
# components variables
a_cim = {'sram':1.7, 'dram': 0.75, 'rram': 105}
e_cim = {'sram': 7.352323, 'dram': 8.75, 'rram': 7.00}
a_other = {'global_buffer': 1756.8, 'pe_buffer': 878.9, 'accumulator': 123.78, 'router': 16.306, 'others': 49.84, 'periphery':173.62}
e_other = {'buffer': 87.8233, 'accumulator': 1.6, 'router': 0.03188, 'others': 1.1181, 'periphery':115.94544}

# computation variables
a_mac = {'8b': 135.1*pow(10, -6), '16b': 2, '32b': 3, '64b': 4}
t_mac = {'8b': 18.61174, '16b': 18.61174, '32b': 3, '64b': 24.8842}
e_mac = {'8b': 0.024, '16b': 7.352323, '32b': 3, '64b': 0.04}

# communication variables
# we suppose 1G hz for out system ,so 41 cycles = 41 ns
# one package in the network is 1500 B
package_size = 1500
t_package = 41
t_recv_send = 2
t_trans_per_bit = 21
e_trans_per_bit = 1.17
# 1ps for 0.1 mm's link
delay_per_hop = 10 * 0.001
# inter-pe bandwidth constraints
max_bandwidth = 10
inter_tile_cost = 30
