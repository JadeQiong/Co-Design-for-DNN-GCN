# HARDWARE
# time: ns, energy: pJ/bit, area: mm^2
# components variables
a_cim = {}
e_cim = {}
a_other = {'router':6.306, 'sram':153.241, 'dram':1, 'rram':10.5}
e_other = {'router':3.388, 'sram':1, 'dram':9.75, 'rram':1}

# computation variables
a_mac = {'8b':135.1*pow(10,-6),'16b':2,'32b':3,'64b':4}
t_mac = {'8b':1,'16b':2,'32b':3,'64b':4}
e_mac = {'8b':0.024,'16b':2,'32b':3,'64b':4}

# communication variables
t_package = 41
t_recv_send = 2
t_trans = 21
e_trans = 1.17
