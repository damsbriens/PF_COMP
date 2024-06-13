from torch import square, tensor, float
import torch
from torch.nn.functional import relu, l1_loss
#import torch_geometric
import numpy as np
#import matplotlib.pyplot as plt
#import time
#import copy
import pandas as pds
from utils import compute_eq_loss, compute_ineq_losses


#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def init_lambda(loader,characteristics, device, batch_size): # We make sure the size of the lambdas fit the size of the loss dictionnaries
    '''
    This is where we initiate the values of the AL multipliers to be 0. Most importantly we decide of the shape of those tensors here.
    Loader here is the training set.
    '''
    node_nb = characteristics['node_nb']
    edge_nb = len(characteristics['actual_edges'])
    gen_nb = len(characteristics['gen_index'])
    nb_of_batches = len(loader)
    lambdas = {}
    lambdas['lf_eq'] = torch.zeros((nb_of_batches, 2, node_nb ), device=device) # 2 equality constraints per node per batch
    lambdas['lh_ineq_node'] = torch.zeros((nb_of_batches, 2, node_nb), device= device)

    return lambdas



#"""
def training_step_coo_lagr(loader, characteristics, model, optimizer, previous_loss, lambdas, penalty_multipliers, run, device):
    '''
    This is where the magic happens. This is the main function that is being called at every epoch. It recieves the loss and the AL multiplers of the previous iteration as an input
    This function is adapted for multiple inputs.
    '''

    norm_coeffs = characteristics['norm_coeffs']
    nb_nodes = characteristics['node_nb']

    edge_from_bus = characteristics["edge_to_bus"][0][0].to(device)
    edge_to_bus = characteristics["edge_to_bus"][0][1].to(device)
    gen_to_bus = characteristics["gen_to_bus"].to(device)
    edge_nb = len(characteristics['actual_edges'])
    base_MVA = characteristics['norm_coeffs']['Base_MVA']
    gen_index = torch.tensor(characteristics['gen_index']).to(device)
    G = torch.tensor(characteristics['G']).unsqueeze(1).to(device)
    B = torch.tensor(characteristics['B']).unsqueeze(1).to(device)
    gen_nb = len(characteristics['gen_index'])
    S_P = characteristics['S_P']
    SG_Q = characteristics['SG_Q']
    NG_V = characteristics['NG_V']


    # adaptive parameter update
    muf = penalty_multipliers['muf']
    muh = penalty_multipliers['muh']
    
    previous_loss = previous_loss.copy() #This previous loss dictionnary has the same shape as the AL multiplers
    #We have to copy to avoid torch issues with in place operations being changed

    nb_batches = len(loader)

    # Stuff to log in later
    eq_loss_avg = 0
    al_eq_loss_avg = 0
    node_ineq_loss_avg = 0

    for i, data in enumerate(loader):
        optimizer.zero_grad()
        model.eval() #Set the model to evaluation mode to compute the losses
        batch_size = data.num_graphs

        # Step 1: Get the Augmented Lagrangian multipliers ------------------------------------------------------------------------------------------------------------------------------

        # Update all of the multipliers
        lambdas['lf_eq'][i] = (lambdas['lf_eq'][i] + 2*muf*previous_loss['eq_loss'][i]).to(device) #We have to update them with the loss of this batch at the previous iteration
        lambdas['lh_ineq_node'][i] = (lambdas['lh_ineq_node'][i] + 2*muh*previous_loss['node_ineq_loss'][i]).to(device)

        # Step 2: Compute the losses for the current batch -------------------------------------------------------------------------------------------------------------------------------
        data.to(device)
        #print(data.x)
        output = model(data)

        # Denormalize the Outputs to properly compute the losses
        p_in = data.x[:, 0]*norm_coeffs['P_norm']
        q_in = data.x[:, 1]*norm_coeffs['Q_norm']
        V_in = data.x[:, 2]*norm_coeffs['V_norm']
        #print(p_in)

        V_in = V_in.view(batch_size, nb_nodes).T[:nb_nodes].to(device)
        p_in = p_in.view(batch_size, nb_nodes).T.to(device)
        q_in = q_in.view(batch_size, nb_nodes).T.to(device)

        p_out = output[:, 0]*norm_coeffs['P_norm']
        q_out = output[:, 1]*norm_coeffs['Q_norm']
        V_out = output[:,2]*norm_coeffs['V_norm']
        Theta = output[:,3]*(np.pi/180)*norm_coeffs['Theta_norm']


        Theta = Theta.view(batch_size, nb_nodes).T[:nb_nodes].to(device)
        V_out = V_out.view(batch_size, nb_nodes).T[:nb_nodes].to(device)
        p_out = p_out.view(batch_size, nb_nodes).T.to(device)
        q_out = q_out.view(batch_size, nb_nodes).T.to(device)
        Theta[characteristics['Ref_node'],:] = 0

        #We only look at the relevant quantities. The others are set to 0
        V_out = NG_V @ V_out
        p_out = S_P @ p_out
        q_out = SG_Q @ q_out

        V = V_out + V_in

        #Get the losses from the utils.py file
        eq_loss = compute_eq_loss(p_in,q_in, V, p_out,q_out, Theta, edge_from_bus, edge_to_bus, gen_to_bus, base_MVA, G, B, batch_size, device)
        #print(eq_loss)
        #Pg = (gen_to_bus @ pg).to(device)
        #plate_loss = torch.abs(pd.sum() - Pg.sum())
        
        node_ineq_loss = compute_ineq_losses(V, p_out, nb_nodes, batch_size, device)
        #Equality losses:
        st_eq_loss = square(eq_loss).sum().to(device)
        al_eq_loss = (eq_loss*lambdas['lf_eq'][i]).sum().to(device)
        

        # Generator Inequality constraints - all nodes (V_min <= V <= V_max)
        st_node_ineq_loss = square(node_ineq_loss).sum().to(device)
        al_node_ineq_loss = (node_ineq_loss*lambdas['lh_ineq_node'][i]).sum().to(device)


        #Store the computed loss for the next epoch AL multiplier computation in the previous_loss dictionnary
        previous_loss['eq_loss'][i] = eq_loss.detach().clone()
        previous_loss['node_ineq_loss'][i] = node_ineq_loss.detach().clone()

        # Step 3: Assemble all of the losses ------------------------------------------------------------------------------------------------------------------------------------------

        #This is for the full AC-OPF
        non_cost_loss =   muf * (st_eq_loss) + muh * (st_node_ineq_loss) + al_node_ineq_loss + al_eq_loss

        #This is for the Economic Dispatch test
        #non_cost_loss = muh * (st_gen_ineq_loss + st_node_ineq_loss + plate_loss) + al_gen_ineq_loss + al_node_ineq_loss #+ al_plate_loss
        #non_cost_loss = muh * (st_ineq_loss) + al_ineq_loss + plate_loss
        #non_cost_loss = muh*st_plate_loss + al_plate_loss
        #loss = non_cost_loss
        #cost_loss = res_cost(pg, qg,characteristics, batch_size)
        loss = (non_cost_loss).to(device) #You can add an extra scaling to the cost on Q (cost_loss[1]), 3* is what was done for the tuned version of Case9Q with M = 1

        # Step 4: Update the nodes through gradient descent ---------------------------------------------------------------------------------------------------------------------------
        model.train()  #Once the losses are computed we switch to train mode in order to update the model
        loss.backward(retain_graph=True)
        optimizer.step()    

        # Step 5: Log all of the losses in Neptune or results_saver ------------------------------------------------------------------------------------------------------------------------------------
        stats = []

        eq_loss_avg += abs(eq_loss).sum().detach()#.cpu()
        al_eq_loss_avg += al_eq_loss.sum().detach()#.cpu()
        node_ineq_loss_avg += node_ineq_loss.sum().detach()#.cpu()
        #cost_loss_avg += total_costs.sum().detach()#.cpu()

    if run != 0:
        run["aug_l/node_ineq"].log(node_ineq_loss_avg/nb_batches)
        run["aug_l/eq"].log(eq_loss_avg/nb_batches)
        run["aug_l/al_eq"].log(al_eq_loss_avg/nb_batches)
        #run["aug_l/al_flow"].log(al_flow_loss_avg/nb_batches)
        #run["train/true cost"].log(cost_loss_avg/nb_batches)

        #current_lr = optimizer.param_groups[0]['lr']
        #run["learning rate"].append(current_lr)
    
    # Step 5: Update the penalty pultipliers ------------------------------------------------------------------------------------------------------------------------------------------
    
    penalty_multipliers["muh"] *= penalty_multipliers["betah"]
    penalty_multipliers["muf"] *= penalty_multipliers["betaf"]
    penalty_multipliers["mun"] *= penalty_multipliers["betan"]

    return penalty_multipliers, previous_loss, stats
#"""

def get_zero_constraints(model, loader,characteristics, device, batch_size):
    '''
    Obsolete. Was the initialization function back when I used get_hard_constraints() to compute the loss.
    '''
    node_nb = characteristics['node_nb']
    actual_edges = characteristics['actual_edges']
    edge_nb = len(actual_edges)
    gen_nb = len(characteristics['gen_index'])

    #Important to keep in mind what the dimensions of all the loss tensors are --------------------------------------------------------------
    eq_loss_vec = torch.zeros((len(loader),2,node_nb), device=device) # (nb of batches , 2 , nb of nodes)
    gen_ineq_loss_vec = torch.zeros((len(loader),4, gen_nb), device=device) # (nb of batches , 4 , nb of gens)
    node_ineq_loss_vec = torch.zeros((len(loader),2, node_nb), device=device) # (nb of batches , 2 , nb of nodes)
    flow_loss_vec = torch.zeros((len(loader),edge_nb), device=device) # (nb of batches , nb of edges)
    # ---------------------------------------------------------------------------------------------------------------------------------------
    return {'eq_loss': eq_loss_vec,'gen_ineq_loss': gen_ineq_loss_vec,'node_ineq_loss': node_ineq_loss_vec, 'flow_loss': flow_loss_vec}
