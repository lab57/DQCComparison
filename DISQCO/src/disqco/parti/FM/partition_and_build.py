from disqco.parti.FM.FM_hetero import run_FM_hetero_dummy
from disqco.parti.FM.FM_methods import set_initial_sub_partitions, order_nodes, map_assignment
from disqco.graphs.GCP_hypergraph import SubGraphManager
from disqco.parti.FM.multilevel_FM import MLFM_recursive_hetero_mapped
from disqco.parti.FM.FM_methods import calculate_full_cost_hetero,get_all_configs, get_all_costs_hetero
import numpy as np
import matplotlib.pyplot as plt
from disqco.drawing.tikz_drawing import draw_graph_tikz


    
def partition_and_build_subgraphs(subgraph, assignment, qpu_sizes, num_partitions, limit, max_gain, passes, stochastic, active_nodes, log, add_initial, costs, network, node_map, assignment_map, dummy_nodes, node_list, level, network_level_list, sub_graph_manager,subgraph_list,sub_assignments,index):


    if costs is None and len(node_map) <= 12:
        configs = get_all_configs(len(node_map), hetero=True)
        costs, edge_tree = get_all_costs_hetero(network, configs, node_map=node_map)
    #     print("Found costs")
    else:
        costs = {}
        
    # print("Node map:", node_map)
    # for key, cost in costs.items():
    #     print(f"Cost for {key}: {cost}")

    # costs = {}

    assignment_list_coarse, cost_list_coarse, _ = MLFM_recursive_hetero_mapped(
        graph = subgraph,
        initial_assignment=assignment,
        qpu_info=qpu_sizes,
        network=network,
        limit = 'qubit',
        pass_list=[50]*6 + [1]*20,
        stochastic=True,
        node_map=node_map,
        costs=costs,
        assignment_map=assignment_map,
        dummy_nodes=dummy_nodes,
        log=False,
        level_limit=None,
        node_list=node_list,
    )

    final_assignment_sub = assignment_list_coarse[np.argmin(cost_list_coarse)]
    sub_assignments.append(final_assignment_sub)
                                                                
    if level != len(network_level_list)-1:
        next_nets = network_level_list[level+1][index:]
        subgraphs = sub_graph_manager.build_partition_subgraphs(subgraph, final_assignment_sub, num_partitions, node_map = node_map, current_network=network, new_networks=next_nets, assignment_map=assignment_map, dummy_nodes=dummy_nodes)
    else:
        subgraphs = []
        

    return sub_assignments, subgraphs
