from torch import square, relu, zeros, tensor, sin, cos, sqrt, float
import torch
from torch.nn.functional import relu, l1_loss
#import torch_geometric
#import numpy as np
#import matplotlib.pyplot as plt
#import time
import pandas as pds
import numpy as np


def compute_eq_loss(pg,qg,pd,qd, V, Theta, edge_from_bus, edge_to_bus, gen_to_bus, nb_nodes, edge_nb, base_MVA, edge_limits, G, B, batch_size, device):    

    # Make sure everything is sent to the device
    pg.to(device)
    qg.to(device)
    pd.to(device)
    qd.to(device)
    V.to(device)
    Theta.to(device)
    edge_from_bus.to(device)
    edge_to_bus.to(device)
    gen_to_bus.to(device)
    G = G.to(device)
    B = B.to(device)

    #print(Theta)
    #print(edge_from_bus @ Theta)
    
    Theta_diff = edge_from_bus @ Theta - edge_to_bus @ Theta
    #print(V)
    V_from = edge_from_bus @ V
    V_to = edge_to_bus @ V
    #print(V_from)
    tap = torch.tensor(edge_limits[2]).unsqueeze(1).to(device)
    shunts = torch.tensor(edge_limits[1]).unsqueeze(1).to(device)
    S_ij = torch.tensor(edge_limits[0]).unsqueeze(1).repeat(1,batch_size).to(device)
    #print(square(V_from) * G.unsqueeze(1) * tap.unsqueeze(1))
    #Power Flows
    p_f = base_MVA * (
    square(V_from) * G * tap
    - V_from * V_to * G * torch.cos(Theta_diff)
    - V_from * V_to * B * torch.sin(Theta_diff)
    ).to(device)
    #print(p_f)
    q_f = base_MVA * (
    -square(V_from) * (B + shunts / 2) * tap
    + V_from * V_to * B * torch.cos(Theta_diff)
    - V_from * V_to * G * torch.sin(Theta_diff)
    ).to(device)

    flow_loss = torch.relu(torch.square(p_f) + torch.square(q_f) - torch.square(S_ij)).to(device)
    flow_loss = flow_loss.mean(dim=1).to(device)

    #Equalities
    Pg = (gen_to_bus @ pg).to(device)
    Qg = (gen_to_bus @ qg).to(device)
    #print(Pg)
    res_pg = (edge_from_bus.T @ p_f.to(torch.float32)).to(device)
    res_qg = (edge_from_bus.T @ q_f.to(torch.float32)).to(device)
    #print(res_pg.size())
    #print(pd.size())
    #print(Pg.size())
    res_pg = torch.abs(res_pg + pd - Pg).to(device)
    res_pg = res_pg.mean(dim=1).to(device)
    res_qg = torch.abs(res_qg + qd - Qg).to(device)
    res_qg = res_qg.mean(dim=1).to(device)

    eq_loss = torch.stack((res_pg,res_qg), dim = 0).to(device)

    return eq_loss, flow_loss

def compute_ineq_losses(pg, qg, V, node_limits, gen_index, nb_nodes, gen_nb, batch_size, device):
    #Inequalities

    P_max = torch.index_select(torch.tensor(node_limits['P_max']).to(device), dim=0, index = gen_index).unsqueeze(1).repeat(1,batch_size).to(device)
    P_min = torch.index_select(torch.tensor(node_limits['P_min']).to(device), dim=0, index = gen_index).unsqueeze(1).repeat(1,batch_size).to(device)
    Q_max = torch.index_select(torch.tensor(node_limits['Q_max']).to(device), dim=0, index = gen_index).unsqueeze(1).repeat(1,batch_size).to(device)
    Q_min = torch.index_select(torch.tensor(node_limits['Q_min']).to(device), dim=0, index = gen_index).unsqueeze(1).repeat(1,batch_size).to(device)

    gen_pg = torch.index_select(pg, dim=0, index = gen_index).to(device)
    gen_qg = torch.index_select(qg, dim=0, index = gen_index).to(device)

    P_max_loss = torch.relu(gen_pg - P_max).to(device)
    P_max_loss = P_max_loss.mean(dim=1).to(device)

    P_min_loss = torch.relu(P_min - gen_pg).to(device)
    P_min_loss = P_min_loss.mean(dim=1).to(device)

    Q_max_loss = torch.relu(gen_qg - Q_max).to(device)
    Q_max_loss = Q_max_loss.mean(dim=1).to(device)

    Q_min_loss = torch.relu(Q_min - gen_qg).to(device)
    Q_min_loss = Q_min_loss.mean(dim=1).to(device)
    

    gen_ineq_loss = torch.stack((P_max_loss, P_min_loss, Q_max_loss, Q_min_loss), dim = 0).to(device)

    V_max = torch.tensor(node_limits['V_max'][:nb_nodes]).to(device).unsqueeze(1).repeat(1,batch_size).to(device)
    V_min = torch.tensor(node_limits['V_min'][:nb_nodes]).to(device).unsqueeze(1).repeat(1,batch_size).to(device)

    V_max_loss = (torch.relu(V - V_max)*100).to(device)
    V_max_loss = V_max_loss.mean(dim=1).to(device)

    V_min_loss = (torch.relu(V_min - V)*100).to(device)
    V_min_loss = V_min_loss.mean(dim=1).to(device)

    node_ineq_loss = torch.stack((V_max_loss, V_min_loss), dim = 0).to(device)
    
    return gen_ineq_loss, node_ineq_loss

def compute_total_cost(pg,qg, gen_index, node_chars, batch_size, device, static_costs): #TODO: modify this for several batches
    #Costs
    #print(pg)
    node_chars = torch.index_select(node_chars, dim = 1, index = gen_index).to(device)
    cost_index = torch.tensor([1,2,4,5]).to(device) #This selects cp_1, cp_2, cq_1 and cq_2
    #static_costs_index = torch.tensor([3,6]).to(device)
    #cost_index = torch.tensor([1,2,3,4,5,6]).to(device)
    gen_pg = torch.index_select(pg, dim=0, index = gen_index).to(device)
    gen_qg = torch.index_select(qg, dim=0, index = gen_index).to(device)
    #print(gen_pg)
    costs = torch.index_select(node_chars, dim = 0, index = cost_index).to(torch.float32).to(device)
    #static_costs = torch.index_select(node_chars, dim = 0, index = static_costs_index).to(torch.float32).to(device)
    #static_costs = torch.full((1,batch_size), static_costs.sum()).squeeze(0)
    
    total_costs = (costs[0] @ torch.square(gen_pg) + costs[1] @ gen_pg + 3*costs[2] @ torch.square(gen_qg) + 3*costs[3] @ gen_qg).to(device)
    #total_costs = (costs[0] @ torch.square(gen_pg) + costs[1] @ gen_pg + costs[2] + 3*costs[3] @ torch.square(gen_qg) + 3*costs[4] @ gen_qg).to(device) + costs[5]
    #print(total_costs)
    #print(static_costs)
    #print(total_costs + static_costs)
    return (total_costs + static_costs).mean(dim=0).to(device)

def penalty_multipliers_init(mun0, muf0, muh0, betan, betaf, betah):
    '''
    Initialize the penalty_multipler dictionnary
    '''
    penalty_multipliers = {}
    penalty_multipliers['mun'] = mun0
    penalty_multipliers['muf'] = muf0
    penalty_multipliers['muh'] = muh0
    penalty_multipliers['betan'] = betan
    penalty_multipliers['betaf'] = betaf
    penalty_multipliers['betah'] = betah
    return penalty_multipliers


def compute_flows(V, Theta, edge_from_bus, edge_to_bus, base_MVA, edge_limits, G, B, batch_size, device):    

    
    Theta_diff = edge_from_bus @ Theta - edge_to_bus @ Theta
    V_from = edge_from_bus @ V
    V_to = edge_to_bus @ V
    tap = torch.tensor(edge_limits[2]).unsqueeze(1).to(device)
    shunts = torch.tensor(edge_limits[1]).unsqueeze(1).to(device)
    S_ij = torch.tensor(edge_limits[0]).unsqueeze(1).repeat(1,batch_size).to(device)

    #Power Flows
    p_f = base_MVA * (
    square(V_from) * G * tap
    - V_from * V_to * G * torch.cos(Theta_diff)
    - V_from * V_to * B * torch.sin(Theta_diff)
    ).to(device)

    q_f = base_MVA * (
    -square(V_from) * (B + shunts / 2) * tap
    + V_from * V_to * B * torch.cos(Theta_diff)
    - V_from * V_to * G * torch.sin(Theta_diff)
    ).to(device)
    return p_f,q_f