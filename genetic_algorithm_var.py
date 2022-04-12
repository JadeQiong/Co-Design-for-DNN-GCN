int_code_len = 5
topo_code_len = 10
crossover_rates = {"pe_num": 0.5, "pe_size": 0.5, "pe_numX": 0.2, "pe_numY": 0.2, "global_buffer_size": 0.3, "topo": 0.2}
acc_gene_type = []
for key in crossover_rates.keys():
    acc_gene_type.append(key)
net_gene_type = ['layer_num', 'layers', 'layer_connection']
