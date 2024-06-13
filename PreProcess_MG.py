import numpy as np
import torch
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
import pandas as pd
import pickle
from matpower import start_instance
import random
from time import time

m = start_instance()

# General info:

#case = 'case30'
case = "case14"

points = 1
batch = 1
split = [0.8,0.1,0.1]
mpc =  m.loadcase(case)

Ybus = m.makeYbus(mpc['baseMVA'],mpc['bus'],mpc['branch'])
edge_index = [list(Ybus.indices), sorted(Ybus.indices)]

# Step 1: Edges ------------------------------------------------------------------------------------------------

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
    actual_edges.append([int(line[1])-1,int(line[0])-1])
    edge_limits.append([line[5], line[4], tap]) #S_limit, shunt, tap

    edge_to_bus[1][edge_counter + 1][int(line[0])-1] = 1
    edge_to_bus[0][edge_counter + 1][int(line[1])-1] = 1

    edge_counter += 2

# So far this only counts the original branches with self loops
edge_to_bus = torch.from_numpy(edge_to_bus).type(torch.float32),



#Case14 Specifics
case_loads = [1,2,3,4,5,8,9,10,11,12,13]
case_prods = [1,2,5,5,7,0]
case_v_correction = 5
case_slack = 6

#Useful matrices for PF computations - all diagnoal matrices:
S_P = torch.zeros((nb_nodes, nb_nodes)) #Slack bus for P_out
SG_Q = torch.zeros((nb_nodes, nb_nodes)) #Slack and Gen for Q_out
NG_V = torch.zeros((nb_nodes, nb_nodes)) # Non-Generator and non-slack for V

S_P[case_slack][case_slack] = 1
SG_Q[case_slack][case_slack] = 1

for i in case_prods:
    SG_Q[i][i] = 1

for j in range(nb_nodes):
    if j not in case_prods:
        NG_V[j][j] = 1

NG_V[case_slack][case_slack] = 0

types = torch.zeros(nb_nodes)

path = 'input_data_local\\lips_case14_sandbox\\Benchmark_competition\\val\\'
Ybus = np.load(path + 'Ybus.npz')

# We start by reformatting the generators
prod_p = torch.tensor(np.load(path + 'prod_p.npz')['data'][0])
prod_v = torch.tensor(np.load(path + 'prod_v.npz')['data'][0])
len_prods = len(prod_p)

prods_to_bus = np.zeros((nb_nodes,len_prods))
for m in range(len(case_prods)):
    prods_to_bus[case_prods[m]][m] = 1
    types[case_prods[m]] = 1

prods_to_bus = torch.from_numpy(prods_to_bus).type(torch.float32) #Same as the old gen_to_bus

#print(prod_v)

#print(prods_to_bus)
prod_p = prods_to_bus @ prod_p
prod_v = prods_to_bus @ prod_v
prod_v[case_v_correction] *= 0.5


# We next move on to the loads
load_p = torch.tensor(np.load(path + 'load_p.npz')['data'][0])
load_q = torch.tensor(np.load(path + 'load_q.npz')['data'][0])
len_loads = len(load_p)

#print(load_q)
#print(len(case_14_loads))
#print(len())

load_to_bus = np.zeros((nb_nodes,len_loads))
for m in range(len(case_loads)):
    load_to_bus[case_loads[m]][m] = 1
    types[case_loads[m]] += 2

load_to_bus = torch.from_numpy(load_to_bus).type(torch.float32)
#print(load_to_bus)
load_p = load_to_bus @ load_p
load_q = load_to_bus @ load_q

#Finally we take case of the slack bus
#Theta = torch.zeros(nb_nodes)

slack = torch.tensor(np.load(path + 'slack.npz')['data'][0])
prod_v[case_slack] = slack[1]


# Everything is brought together
net_p = prod_p - load_p
index = torch.arange(nb_nodes) * 0.01
hash = torch.full((nb_nodes,), 0.0002 * prod_p.sum())

#print(prod_v)

V_norm = 1000 #Structure here is not efficient. Is kept this way in case we want to change norms
P_norm = 200
Q_norm = 100
norm_coeffs = {'V_norm': V_norm, 'P_norm': P_norm, 'Q_norm': Q_norm, 'Theta_norm': 360., 'Base_MVA': Base_MVA}





input = torch.stack([net_p/P_norm, load_q/Q_norm ,prod_v/V_norm, types*0.1, index, hash]).T

#print(types)

inputs = []#torch.zeros((points, nb_nodes, 6))
inputs.append([input])
#print(inputs)

edge_index = torch.tensor(edge_index, dtype=torch.long)

#"""
#Step 7: Distribution of points into leaders --------------------------------------------------------------------------

data_list = []
device = 'cpu'
X = {'X': inputs}
#Y = {'Y': node_optimum}
for i in range(len(X['X'])):
    N = torch.tensor(X['X'][i][0], dtype=torch.float, device=device)
    #Y_o = torch.tensor(Y['Y'][i][0], dtype=torch.float, device=device)
    data = Data(x=N, edge_index=edge_index.to(device), edge_attr= None).to(device)#, y=Y_o).to(device)  #Since all of the rest of the information is shared between all points, no reason to include it in each data point
    print(data)
    data_list.append(data)  #we removed the .to(device) from the edge_index

train_set, val_set, test_set = torch.utils.data.random_split(data_list, [0.8, 0.1, 0.1]) #This doesn't work
print(len(train_set))
print(len(val_set))
print(len(test_set))

if points >= 10:
    print('case')
    train_loader = DataLoader(train_set, batch_size=batch, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=batch, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch, shuffle=False)

if points < 10:
    print('Train, Validation and Test set will be the same')
    train_loader = DataLoader(data_list, batch_size=batch, shuffle=False)
    val_loader = DataLoader(data_list, batch_size=batch, shuffle=False)
    test_loader = DataLoader(data_list, batch_size=1, shuffle=False)



characteristics = {'node_nb': nb_nodes , 'edge_limits': edge_limits, 'actual_edges': actual_edges, 'G': G, 'B': B, 'norm_coeffs': norm_coeffs, 'Ref_node': case_slack , 'gen_index': case_prods, 'gen_to_bus': prods_to_bus, 'edge_to_bus': edge_to_bus, 'S_P': S_P, 'SG_Q': SG_Q, 'NG_V': NG_V}
characteristics['static_cost'] = mpc['gencost'].T[6].sum()

print('Saving {}_{}_{}'.format(case, points, batch))
torch.save(train_loader, "Input_Data/train_loader_{}_{}_{}.pt".format(case, points, batch))
torch.save(val_loader, "Input_Data/val_loader_{}_{}_{}.pt".format(case, points, batch))
torch.save(test_loader, "Input_Data/test_loader_{}_{}_{}.pt".format(case, points, batch))
#print(characteristics)
with open('Input_Data/characteristics_{}.pkl'.format(case), 'wb') as f:
    pickle.dump(characteristics, f) 

def return_func():
    return train_loader, val_loader, test_loader, characteristics

#"""