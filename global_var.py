# HARDWARE
# time: ns, energy: pJ/bit, area: um^2
# components variables
a_cim = {'sram': 153.241, 'dram': 1, 'rram': 10.5}
e_cim = {'sram': 6.94085, 'dram': 9.75, 'rram': 1}
a_other = {'global_buffer': 1756.8, 'pe_buffer': 878.9, 'accumulator': 123.78, 'router': 16.306, 'others': 499.84}
e_other = {'buffer': 87.8233, 'accumulator': 1.6, 'router': 3.388, 'others': 7.1181}

# computation variables
a_mac = {'8b': 135.1*pow(10, -6), '16b': 2, '32b': 3, '64b': 4}
t_mac = {'8b': 12.4421, '16b': 2, '32b': 3, '64b': 4}
e_mac = {'8b': 0.024, '16b': 2, '32b': 3, '64b': 4}

# communication variables
t_package = 41
t_recv_send = 2
t_trans = 21
e_trans = 1.17

# inter-pe bandwidth constraints
max_bandwidth = 10
