int_code_len = 5
topo_code_len = 30
crossover_rates = {"pe_num": 0.6, "pe_size": 1, "pe_numX": 0.8, "pe_numY": 0.8, "global_buffer_size": 0.8, "topo": 0.2}
mutate_rates = {"pe_size": 0.2, "pe_numX": 0.2, "pe_numY": 0.2, "global_buffer_size": 0.1, "topo": 0.2}
acc_gene_type = []
for key in crossover_rates.keys():
    acc_gene_type.append(key)
net_gene_type = ['layer_num', 'layers', 'layer_connection']

