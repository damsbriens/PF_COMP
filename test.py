#print("Hello World!")

import numpy as np
#import torch
path = 'input_data_local\\lips_case14_sandbox\\Benchmark_competition\\val\\'

data_type = 'Ybus'
data = np.load(path + data_type +'.npz')
#print(data.keys())
#keys_list = list(data.keys())
#print(keys_list)
#for key in keys_list:
#    print(f"Array {key}:", data[key])
#print(data['data'].shape)
#print(data['data'][0][1])
#print(data['data'][0][4])
#print(data['data'][0][15])
#print(data['data'][0][3])
#print(data['data'][0][6])
#print(data['data'][0][3][8])
#print(data['data'][0][4][5])
#print()
#for i in range(50):
    #print(data['data'][i][3][6])

#print(data['data'][0][12][19])

edges = [[],[]]
actual_edges_1 = []
for i in range(0,28):
    for j in range(0,i):
    #for j in range(0,28):
        #print(data['data'][0][i][j])
        #if (i == j):
        #    edges[0].append(i)
         #   edges[1].append(j)
        if (data['data'][14][i][j] != 0j) and i != j:
            #print('recorded')
            #edges[0].append(i)
            #edges[1].append(j)

            #edges[0].append(j)
            #edges[1].append(i)

            #actual_edges_1.append([i,j])
            if i > 13:
                actual_edges_1.append([i - 14,j])
            else:
                actual_edges_1.append([j,i])

data_type_2 = 'line_status'
data_2 = np.load('input_data_local\\lips_case14_sandbox\\Benchmark_competition\\train\\'+ data_type_2 +'.npz')
#print(edges)
#print(len(edges[0]))
#print(actual_edges_1)
#print(len(actual_edges_1))
#print(data_2['data'][14])
#print(20 - len(actual_edges_1)/2)
nb_nodes = 14
load_to_node = np.zeros((nb_nodes, len(data['data'][0])))

#False -4 is 3-8
#False 1 is 0-4
#False 7 is 5 - 12

"""

import numpy as np
import torch
import pandas as pd
import pickle
from matpower import start_instance

m = start_instance()

case = "case14"

mpc =  m.loadcase(case)

Ybus = m.makeYbus(mpc['baseMVA'],mpc['bus'],mpc['branch'])
edge_index = [list(Ybus.indices), sorted(Ybus.indices)]

nb_nodes = len(mpc['bus'])
nb_edges = len(mpc['branch'])
nb_gens = len(mpc['gen'])
Base_MVA = mpc['baseMVA']


edge_limits = []
actual_edges = []
G = []
B = []
edge_to_bus = np.zeros((2, 2*nb_edges, nb_nodes)) #Can change this to 2,nb_edges
edge_counter = 0
for line in mpc['branch']:
    if line[8] == 0:
        tap = 1
    else:
        tap = line[8]
    Y = 1/((line[2] + 1j * line[3])*tap)

    if line[5] == 0: #Case where there is no transmission line limit
        line[5] = 10000
    G.append(np.real(Y))
    B.append(np.imag(Y))
    actual_edges.append([int(line[0])-1,int(line[1])-1])
    edge_limits.append([line[5], line[4], 1/tap]) #This isn't correct in the original code. It isn't symmetric

    edge_to_bus[0][edge_counter][int(line[0])-1] = 1
    edge_to_bus[1][edge_counter][int(line[1])-1] = 1


    G.append(np.real(Y))
    B.append(np.imag(Y))
    #actual_edges.append([int(line[1])-1,int(line[0])-1])

#print(Ybus)
#print(edge_index[0][27:29])
#print(edge_index[1][27:29])
#print(Ybus.data[27:29])
#print(edge_index)
#print(len(edge_index[0]))
print(actual_edges)
#print(len(actual_edges))
#"""