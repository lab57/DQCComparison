import copy

def space_mapping(qpu_info, num_layers):
    qpu_mapping = {}
    qubit_index = 0
    if isinstance(qpu_info, dict):
        qpu_sizes = list(qpu_info.values())
    else:
        qpu_sizes = qpu_info
    for j, qpu_size in enumerate(qpu_sizes):
        qubit_list = []
        for _ in range(qpu_size):
            qubit_list.append(qubit_index)
            qubit_index += 1
        qpu_mapping[j] = qubit_list
    space_mapping = []
    for t in range(num_layers):
        space_mapping.append(copy.deepcopy(qpu_mapping))
    
    return space_mapping

def get_pos_list(graph, num_qubits, assignment, space_map, assignment_map = None):

    num_layers = len(space_map)
    pos_list = [[None for _ in range(num_qubits)] for _ in range(num_layers)]

    if assignment_map is not None:
        inverse_assignment_map = {}
        for node in assignment_map:
            inverse_assignment_map[assignment_map[node]] = node
    
    for q in range(num_qubits):
        old_partition = None
        for t in range(num_layers):
            if assignment_map is not None:
                q, t = inverse_assignment_map[(q, t)]
            # partition = assignment[(q,t)]
            partition = assignment[t][q]
            if old_partition is not None:
                if partition == old_partition:
                    if y_pos in space_map[t][partition]:
                        y_pos = pos_list[t-1][q]
                        pos_list[t][q] = y_pos 
                        space_map[t][partition].remove(y_pos)
                    else:
                        qubit_list = space_map[t][partition]
                        y_pos = qubit_list.pop(0)
                        pos_list[t][q] = y_pos

                else:
                    qubit_list = space_map[t][partition]
                    y_pos = qubit_list.pop(0)
                    pos_list[t][q] = y_pos
            else:
                qubit_list = space_map[t][partition]
                y_pos = qubit_list.pop(0)
                pos_list[t][q] = y_pos
            old_partition = partition
    return pos_list

def get_pos_list_ext(graph, num_qubits, assignment, space_map, qpu_sizes, assignment_map = None):

    num_layers = len(space_map)
    pos_list = [[None for _ in range(num_qubits)] for _ in range(num_layers)]

    if assignment_map is not None:
        inverse_assignment_map = {}
        for node in assignment_map:
            inverse_assignment_map[assignment_map[node]] = node
    
    for q in range(num_qubits):
        old_partition = None
        for t in range(num_layers):
            if assignment_map is not None:
                q, t = inverse_assignment_map[(q, t)]
            # partition = assignment[(q,t)]
            partition = assignment[t][q]
            if old_partition is not None:
                if partition == old_partition:
                    if y_pos in space_map[t][partition]:
                        y_pos = pos_list[t-1][q]
                        pos_list[t][q] = y_pos 
                        space_map[t][partition].remove(y_pos)
                    else:
                        qubit_list = space_map[t][partition]
                        y_pos = qubit_list.pop(0)
                        pos_list[t][q] = y_pos

                else:
                    qubit_list = space_map[t][partition]
                    y_pos = qubit_list.pop(0)
                    pos_list[t][q] = y_pos
            else:
                qubit_list = space_map[t][partition]
                y_pos = qubit_list.pop(0)
                pos_list[t][q] = y_pos
            old_partition = partition


    # pos_dict = {}
    # for t in range(len(pos_list)):
    #     for q in range(num_qubits):
    #         pos_dict[(q, t)] = pos_list[t][q]

    # for node in graph.nodes():
    #     if node not in pos_dict:
    #         partition = assignment[node]
    #         if partition == 0:
    #             boundary1 = 0
    #             boundary2 = qpu_sizes[0]
    #         else:
    #             boundary1 = qpu_sizes[partition-1]
    #             boundary2 = qpu_sizes[partition]
    #         position = (boundary1 + boundary2) // 2

    #         pos_dict[node] = position


    return pos_list
