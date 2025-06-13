from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit.dagnode import DAGOpNode

def get_reg_mapping(circuit):
    qubit_indeces = {}
    index = 0
    for reg in circuit.qregs:
        for n in range(reg.size):
            qubit_indeces[(reg.name,n)] = index
            index += 1
    return qubit_indeces

def circuit_to_gate_layers(circuit, qpu_sizes = None):
    "Uses qiskit DAG circuit to group gates into sublists by layer/timestep of the circuit"
    dag = circuit_to_dag(circuit)
    layers = list(dag.multigraph_layers())
    print(f"Number of layers: {len(layers)}")
    layer_gates = []
    qubit_mapping = get_reg_mapping(circuit)
    if qpu_sizes is not None:
        max_pairs = find_max_interactions(qpu_sizes)
    for layer in layers:
        pairs = 0
        layer_info = []
        for node in layer:
            if isinstance(node, DAGOpNode):
                gate_name = node.name
                qubits = [qubit_mapping[(qubit._register.name,qubit._index)] for qubit in node.qargs]
                registers = [qubit._register.name for qubit in node.qargs]
                params = node.op.params

                gate_info = [gate_name, qubits, registers, params]
                layer_info.append(gate_info)
                if qpu_sizes is not None:
                    if len(qubits) == 2:
                        pairs += 1
                        if pairs >= max_pairs:
                            layer_gates.append(layer_info)
                            layer_info = []
                            pairs = 0
        if layer_info != []:
            layer_gates.append(layer_info)
    return layer_gates

def find_max_interactions(qpu_info):
    max_pairs_qpu = []
    for n in range(len(qpu_info)):
        if qpu_info[n] % 2 == 1:
            max_pairs_qpu.append((qpu_info[n]-1)//2)
        else:
            max_pairs_qpu.append(qpu_info[n]//2)
    max_pairs = sum(max_pairs_qpu)
    return max_pairs

def layer_list_to_dict(layers):
    d = {}
    for i,layer in enumerate(layers):
        d[i] = []
        for gate in layer:
            gate_dict = {}
            name = gate[0]
            qargs = gate[1]
            qregs = gate[2]
            params = gate[3]
            if gate[0] != 'barrier' and gate[0] != 'measure':
                if len(gate) < 5:
                    if len(qargs) < 2:
                        gate_dict['type'] = 'single-qubit'
                    elif len(qargs) == 2:
                        gate_dict['type'] = 'two-qubit'
                    gate_dict['name'] = name
                    gate_dict['qargs'] = qargs
                    gate_dict['qregs'] = qregs
                    gate_dict['params'] = params
                else:
                    gate_dict['type'] = 'group'
                    gate_dict['root'] = qargs[0]
                    gate_dict['sub-gates'] = []
                    gate1 = {}
                    gate1['type'] = 'two-qubit'
                    gate1['name'] = name
                    gate1['time'] = i
                    gate1['qargs'] = qargs
                    gate1['qregs'] = qregs
                    gate1['params'] = params
                    gate_dict['sub-gates'].append(gate1)
                    for j in range(5,len(gate)):
                        gate_i_list = gate[j]
                        gate_i = {}
                        if gate_i_list[0] == gate_i_list[1]:
                            gate_i['type'] = 'single-qubit'
                            l = 1
                        else:
                            gate_i['type'] = 'two-qubit'
                            l = 2
                        gate_i['name'] = gate_i_list[-1]
                        gate_i['qargs'] = [gate_i_list[0],gate_i_list[1]]
                        gate_i['qregs'] = ['q' for _ in range(l)]
                        gate_i['params'] = gate_i_list[-2]
                        gate_i['time'] = gate_i_list[2]
                        gate_dict['sub-gates'].append(gate_i)
                d[i].append(gate_dict)
    return d


