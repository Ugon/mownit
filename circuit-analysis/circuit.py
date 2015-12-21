import networkx as nx
import numpy as np
import scipy.linalg as sp
import matplotlib.pyplot as plt
import argparse
import random


def read_multi_graph(filename):
    return nx.read_edgelist(filename, create_using=nx.MultiGraph(), nodetype=int, data=[('R', float)])


def multi_graph_to_regular_graph(multi_graph):
    regular_graph = nx.Graph()
    number_of_nodes = multi_graph.number_of_nodes()
    for (u, v, data) in multi_graph.edges_iter(data=True):
        if (u, v) not in regular_graph.edges() and (v, u) not in regular_graph.edges():
            regular_graph.add_edge(u, v, {'R': data['R']})
        else:
            number_of_nodes += 1
            w = number_of_nodes
            regular_graph.add_node(w, {'auxiliary': True})
            regular_graph.add_edge(u, w, {'R': data['R']})
            regular_graph.add_edge(w, v, {'R': 0})
    return regular_graph


def regular_graph_to_multi_graph(regular_graph):
    multi_graph = nx.Graph()
    auxiliary_nodes = nx.get_node_attributes(regular_graph, 'auxiliary')
    for (u, v, data) in graph.edges_iter(data=True):
        if u not in auxiliary_nodes and v not in auxiliary_nodes:
            multi_graph.add_edge(u, v, data)
    for w in auxiliary_nodes:
        [u, v] = regular_graph.neighbors(w)
        if (u, w) not in regular_graph.edges():
            u, v = v, u
        data = {'R': regular_graph.get_edge_data(u, w)['R'],
                'I': regular_graph.get_edge_data(u, w)['I']}
        if 'E' in (u, w):
            data['E'] = regular_graph.get_edge_data(u, w)['E']
        multi_graph.add_edge(u, v, data)
    return multi_graph


def add_electromotive_force(graph, neg, pos, sem):
    if (neg, pos) not in graph.edges() and (pos, neg) not in graph.edges():
        graph.add_edge(neg, pos, {'E': sem, 'R': float(0)})
    else:
        graph[neg][pos]['E'] = sem


def apply_kirchhoffs_current_law(graph, matrix):
    for k in xrange(graph.number_of_edges()):
        (u, v) = graph.edges()[k]
        matrix[u - 1][k] = 1
        if v < graph.number_of_nodes():  # skip redundant equation (linearly dependent)
            matrix[v - 1][k] = -1
        graph[u][v]['variable_index'] = k


def apply_kirchhoffs_voltage_law(graph, matrix, vector):
    resistances = nx.get_edge_attributes(graph, 'R')
    voltages = nx.get_edge_attributes(graph, 'E')
    k = graph.number_of_nodes() - 1
    for cycle in nx.cycle_basis(graph):
        for j in xrange(len(cycle)):
            u = cycle[j]
            v = cycle[(j + 1) % len(cycle)]
            if u < v:
                coefficient = -1
            else:
                coefficient = 1
                u, v = v, u
            column = graph[u][v]['variable_index']
            matrix[k][column] = resistances[(u, v)] * coefficient
            if (u, v) in voltages:
                vector[k] -= voltages[(u, v)] * coefficient
        k += 1


def calculate_currents(graph):
    number_of_edges = nx.number_of_edges(graph)
    matrix = np.zeros((number_of_edges, number_of_edges))
    vector = np.zeros(number_of_edges)
    apply_kirchhoffs_current_law(graph, matrix)
    apply_kirchhoffs_voltage_law(graph, matrix, vector)
    currents = sp.solve(matrix, vector)
    for (u, v) in graph.edges_iter():
        k = graph[u][v]['variable_index']
        graph[u][v]['I'] = float("%.3g" % currents[k])


def draw_small_circuit(graph):
    edges_labels = dict(
        ((u, v), {key: data[key] for key in ['E', 'I', 'R'] if key in data}) for (u, v, data) in graph.edges(data=True))
    print edges_labels
    pos = nx.spectral_layout(graph)
    nx.draw_networkx(graph, pos=pos, node_size=500)
    nx.draw_networkx_edge_labels(graph, pos=pos, edge_labels=edges_labels, alpha=0.1)
    plt.show()


def draw_large_circuit(graph):
    edges_labels = dict(
        ((u, v), {key: data[key] for key in ['E', 'I', 'R'] if key in data}) for (u, v, data) in graph.edges(data=True))
    print edges_labels
    pos = nx.shell_layout(graph)
    colors = [abs(float(data['I'])) for (u, v, data) in graph.edges(data=True)]
    cmax = max(colors)
    jet = plt.get_cmap('YlOrRd')
    nx.draw_networkx(graph, pos=pos, node_size=0, font_size=20, node_color='g', edge_color=colors, edge_cmap=jet,
                     edge_vmin=0, edge_vmax=cmax, width=3)
    plt.show()


parser = argparse.ArgumentParser()
parser.add_argument('-type', type=str, required=True, choices=['file', 'random'], default='file')
parser.add_argument('-input', type=str, required=False, dest='input', default='defa')
parser.add_argument('-size', type=str, required=True, dest='size', choices=['small', 'large'])
parser.add_argument('-neg', type=int, required=True, dest='neg')
parser.add_argument('-pos', type=int, required=True, dest='pos')
parser.add_argument('-sem', type=float, required=True, dest='sem')
args = vars(parser.parse_args())
t = args['type']
i = args['input']
size = args['size']
neg = args['neg']
pos = args['pos']
sem = args['sem']

if t == 'file' and i == 'defa':
    print "Go, specify a file."
    exit()

if t == 'file':
    graph = read_multi_graph(i)
else:
    # graph = nx.ladder_graph(9)  # set random graph type here
    graph = nx.random_regular_graph(3, 10)  # set random graph type here
    # graph = nx.random_regular_graph(3, 1000)  # set random graph type here
    for (u, v) in graph.edges():
        graph[u][v]['R'] = random.random() * random.randint(0, 1000)

graph = multi_graph_to_regular_graph(graph)
add_electromotive_force(graph, neg, pos, sem)
calculate_currents(graph)
graph = regular_graph_to_multi_graph(graph)

if size == 'small':
    draw_small_circuit(graph)
else:
    draw_large_circuit(graph)

# ./circuit.py -type=random -size=large -neg=4 -pos=8 -sem=10
# ./circuit.py -type=file -input=circuit1.txt -neg=1 -pos=2 -sem=10 -size=small
# ./circuit.py -type=file -input=circuit2.txt -neg=2 -pos=4 -sem=10 -size=small
