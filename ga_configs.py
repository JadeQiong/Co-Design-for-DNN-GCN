int_code_len = 5
topo_code_len = 30
crossover_rates = {"pe_numX": 0.8, "pe_numY": 0.8, "tile_numX": 0.8, "tile_numY": 0.8, "pe_size": 1,
                   "global_buffer_size": 0.8, "pe_topo": 0.2, "tile_topo": 0.2,
                   "chipX": 0.5, "chipY": 0.5, "chiplet_topo": 0.2}
mutate_rates = {"pe_numX": 0.2, "pe_numY": 0.2, "tile_numX": 0.2, "tile_numY": 0.2, "pe_size": 0.2,
                "global_buffer_size": 0.1, "pe_topo": 0.2, "tile_topo": 0.2,
                "chipX": 0.1, "chipY": 0.1, "chiplet_topo": 0.05}
acc_gene_type = ["pe_numX", "pe_numY", "tile_numX", "tile_numY", "pe_size", "global_buffer_size", "pe_topo", "tile_topo"]
chiplet_gene_type = ["chipX", "chipY", "chiplet_topo"]
# python version MATTERS!
# for key in crossover_rates.keys():
#     acc_gene_type.append(key)

net_gene_type = ['layer_num', 'layers', 'layer_connection']