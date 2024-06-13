from torch import square, relu, zeros, tensor, sin, cos, sqrt, float
import torch
from torch.nn.functional import relu, l1_loss
#import torch_geometric
#import numpy as np
#import matplotlib.pyplot as plt
#import time
import pandas as pds
import numpy as np


def compute_eq_loss(p_in,q_in, V, p_out,q_out, Theta, edge_from_bus, edge_to_bus, gen_to_bus, base_MVA, G, B, batch_size, device):  

    # Make sure everything is sent to the device
    p_out.to(device)
    q_out.to(device)
    Theta.to(device)
    edge_from_bus.to(device)
    edge_to_bus.to(device)
    gen_to_bus.to(device)
    G = G.to(device)
    B = B.to(device)
    V = V/100
    #print(p_out)
    #print(V)

    Theta_diff = edge_from_bus @ Theta - edge_to_bus @ Theta
    V_from = edge_from_bus @ V
    V_to = edge_to_bus @ V

    #print(V_from)

    p_f = base_MVA * (
    square(V_from) * G
    - V_from * V_to * G * torch.cos(Theta_diff)
    - V_from * V_to * B * torch.sin(Theta_diff)
    ).to(device)
    #print(p_f)
    q_f = base_MVA * (
    -square(V_from) * (B)
    + V_from * V_to * B * torch.cos(Theta_diff)
    - V_from * V_to * G * torch.sin(Theta_diff)
    ).to(device)

    #print(Pg)
    res_pg = (edge_from_bus.T @ p_f.to(torch.float32)).to(device)
    res_qg = (edge_from_bus.T @ q_f.to(torch.float32)).to(device)
    #print(res_pg.size())
    #print(pd.size())
    #print(Pg.size())
    res_pg = torch.abs(res_pg - p_in - p_out).to(device)
    res_pg = res_pg.mean(dim=1).to(device)
    res_qg = torch.abs(res_qg - q_in - q_out).to(device)
    res_qg = res_qg.mean(dim=1).to(device)

    eq_loss = torch.stack((res_pg,res_qg), dim = 0).to(device)
    #print(res_pg)
    return eq_loss

def compute_ineq_losses(V, p_out, nb_nodes, batch_size, device):
    #Inequalities
    P_min = torch.zeros((batch_size,nb_nodes))

    P_min_loss = (torch.relu(P_min - p_out)).to(device)
    P_min_loss = P_min_loss.mean(dim=1).to(device)

    V_min = torch.ones((batch_size,nb_nodes))

    V_min_loss = (torch.relu(V_min - V)*100).to(device)
    V_min_loss = V_min_loss.mean(dim=1).to(device)
    
    node_ineq_loss = torch.stack((P_min_loss, V_min_loss), dim = 0).to(device)

    return node_ineq_loss



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
