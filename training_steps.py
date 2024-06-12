from torch import square, tensor, float
import torch
from torch.nn.functional import relu, l1_loss
#import torch_geometric
import numpy as np
#import matplotlib.pyplot as plt
#import time
#import copy
import pandas as pds
from utils import compute_eq_loss, compute_ineq_losses, compute_total_cost


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
    lambdas['lh_ineq_gen'] = torch.zeros((nb_of_batches, 4, gen_nb), device= device) # 6 inequality constraints on nodes per batch
    lambdas['lh_ineq_node'] = torch.zeros((nb_of_batches, 2, node_nb), device= device)
    lambdas['lh_flow'] = torch.zeros((nb_of_batches, edge_nb), device=device) # 1 per edge per batch

    return lambdas



#"""
def training_step_coo_lagr(loader, characteristics, model, optimizer, previous_loss, lambdas, penalty_multipliers, run, device, true_costs):
    '''
    This is where the magic happens. This is the main function that is being called at every epoch. It recieves the loss and the AL multiplers of the previous iteration as an input
    This function is adapted for multiple inputs.
    '''
    
    total_loss = 0

    norm_coeffs = characteristics['norm_coeffs']
    nb_nodes = characteristics['node_nb']

    edge_from_bus = characteristics["edge_to_bus"][0][0].to(device)
    edge_to_bus = characteristics["edge_to_bus"][0][1].to(device)
    gen_to_bus = characteristics["gen_to_bus"].to(device)
    edge_nb = len(characteristics['actual_edges'])
    base_MVA = characteristics['norm_coeffs']['Base_MVA']
    edge_limits = characteristics['edge_limits']
    node_limits = characteristics['node_limits']
    gen_index = torch.tensor(characteristics['gen_index']).to(device)
    G = torch.tensor(characteristics['G']).unsqueeze(1).to(device)
    B = torch.tensor(characteristics['B']).unsqueeze(1).to(device)
    gen_nb = len(characteristics['gen_index'])
    node_chars = torch.tensor(characteristics['nodes'].values).to(device)


    # adaptive parameter update
    muf = penalty_multipliers['muf']
    muh = penalty_multipliers['muh']
    mun = penalty_multipliers['mun']
    
    previous_loss = previous_loss.copy() #This previous loss dictionnary has the same shape as the AL multiplers
    #We have to copy to avoid torch issues with in place operations being changed

    nb_batches = len(loader)

    # Stuff to log in later
    eq_loss_avg = 0
    al_eq_loss_avg = 0
    flow_loss_avg = 0
    gen_ineq_loss_avg = 0
    node_ineq_loss_avg = 0
    cost_difference_avg = 0

    for i, data in enumerate(loader):
        optimizer.zero_grad()
        model.eval() #Set the model to evaluation mode to compute the losses
        batch_size = data.num_graphs
        

        # Step 2: Compute the losses for the current batch -------------------------------------------------------------------------------------------------------------------------------
        data.to(device)
        output = model(data)

        # Denormalize the Outputs to properly compute the losses
        pd = data.x[:, 0]*norm_coeffs['P_norm']
        qd = data.x[:, 1]*norm_coeffs['Q_norm']
        pg = output[:, 0]*norm_coeffs['P_norm']
        qg = output[:, 1]*norm_coeffs['Q_norm']
        V = output[:,2]*norm_coeffs['V_norm']
        Theta = output[:,3]*(np.pi/180)*norm_coeffs['Theta_norm']

        Theta = Theta.view(batch_size, nb_nodes).T[:nb_nodes].to(device)
        V = V.view(batch_size, nb_nodes).T[:nb_nodes].to(device)
        pg = pg.view(batch_size, nb_nodes).T.to(device)
        qg = qg.view(batch_size, nb_nodes).T.to(device)
        Theta[characteristics['Ref_node'],:] = 0
        #print(pg)
        pd = pd.view(batch_size, nb_nodes).T[:nb_nodes]
        qd = qd.view(batch_size, nb_nodes).T[:nb_nodes]

        static_costs = torch.full((1,batch_size), characteristics['static_costs'].sum()).squeeze(0)
        true_cost = true_costs[i]
        #print(pd)
        #print(pg)
        #print(V)
        #print(Theta)
        #Theta[characteristics['Ref_node']] = 0.

        #Get the losses from the utils.py file
        eq_loss, flow_loss = compute_eq_loss(pg,qg,pd,qd, V, Theta, edge_from_bus, edge_to_bus, gen_to_bus, nb_nodes, edge_nb, base_MVA, edge_limits, G, B, batch_size, device)
        #print(eq_loss)
        #Pg = (gen_to_bus @ pg).to(device)
        #plate_loss = torch.abs(pd.sum() - Pg.sum())
        
        gen_ineq_loss, node_ineq_loss = compute_ineq_losses(pg,qg,V, node_limits, gen_index, nb_nodes, gen_nb, batch_size, device)

        total_costs = compute_total_cost(pg,qg, gen_index, node_chars, batch_size, device, static_costs)
        #print(total_costs)
        #Equality losses:
        st_eq_loss = square(eq_loss).sum().to(device)
        al_eq_loss = (eq_loss*lambdas['lf_eq'][i]).sum().to(device)
        
        # Generator Inequality constraints - generator nodes (Pg_min <= Pg <= Pg_max ; Qg_min <= Qg <= Qg_max)
        st_gen_ineq_loss = square(gen_ineq_loss).sum().to(device)
        al_gen_ineq_loss = (gen_ineq_loss*lambdas['lh_ineq_gen'][i]).sum().to(device)

        # Generator Inequality constraints - all nodes (V_min <= V <= V_max)
        st_node_ineq_loss = square(node_ineq_loss).sum().to(device)
        al_node_ineq_loss = (node_ineq_loss*lambdas['lh_ineq_node'][i]).sum().to(device)

        # Inequality constraints - Maximum line flow
        flow_loss = flow_loss * 0.001  # Fix done here since this was way to steep. Every time I'd get a huge spike at epoch 40 which broke how the voltages were evolving. This shouldn't be touched too much
        st_flow_loss = square(flow_loss).sum().to(device)
        al_flow_loss = (flow_loss*lambdas['lh_flow'][i]).sum().to(device)


        #Store the computed loss for the next epoch AL multiplier computation in the previous_loss dictionnary
        previous_loss['eq_loss'][i] = eq_loss.detach().clone()
        previous_loss['gen_ineq_loss'][i] = gen_ineq_loss.detach().clone()
        previous_loss['node_ineq_loss'][i] = node_ineq_loss.detach().clone()
        previous_loss['flow_loss'][i] = flow_loss.detach().clone()

        # Step 3: Assemble all of the losses ------------------------------------------------------------------------------------------------------------------------------------------

        #This is for the full AC-OPF
        non_cost_loss =   muf * (st_eq_loss) + muh * (st_gen_ineq_loss + st_node_ineq_loss + st_flow_loss) + al_gen_ineq_loss + al_node_ineq_loss + al_flow_loss + al_eq_loss

        #This is for the Economic Dispatch test
        #non_cost_loss = muh * (st_gen_ineq_loss + st_node_ineq_loss + plate_loss) + al_gen_ineq_loss + al_node_ineq_loss #+ al_plate_loss
        #non_cost_loss = muh * (st_ineq_loss) + al_ineq_loss + plate_loss
        #non_cost_loss = muh*st_plate_loss + al_plate_loss
        #loss = non_cost_loss
        #cost_loss = res_cost(pg, qg,characteristics, batch_size)
        loss = (non_cost_loss + mun * (total_costs)).to(device) #You can add an extra scaling to the cost on Q (cost_loss[1]), 3* is what was done for the tuned version of Case9Q with M = 1

        # Step 4: Update the nodes through gradient descent ---------------------------------------------------------------------------------------------------------------------------
        model.train()  #Once the losses are computed we switch to train mode in order to update the model
        loss.backward(retain_graph=True)
        optimizer.step()    

        # Step 5: Log all of the losses in Neptune or results_saver ------------------------------------------------------------------------------------------------------------------------------------
        stats = []

        eq_loss_avg += abs(eq_loss).sum().detach()#.cpu()
        al_eq_loss_avg += al_eq_loss.sum().detach()#.cpu()
        gen_ineq_loss_avg += gen_ineq_loss.sum().detach()#.cpu()
        node_ineq_loss_avg += node_ineq_loss.sum().detach()#.cpu()
        cost_difference_avg += torch.abs(true_cost - total_costs).detach()
        #cost_loss_avg += total_costs.sum().detach()#.cpu()

        # Step 1: Get the Augmented Lagrangian multipliers ------------------------------------------------------------------------------------------------------------------------------

        # Update all of the multipliers
        lambdas['lf_eq'][i] = (lambdas['lf_eq'][i] + 2*muf*previous_loss['eq_loss'][i]).to(device) #We have to update them with the loss of this batch at the previous iteration
        lambdas['lh_ineq_gen'][i] = (lambdas['lh_ineq_gen'][i] + 2*muh*previous_loss['gen_ineq_loss'][i]).to(device)
        lambdas['lh_ineq_node'][i] = (lambdas['lh_ineq_node'][i] + 2*muh*previous_loss['node_ineq_loss'][i]).to(device)
        lambdas['lh_flow'][i] = (lambdas['lh_flow'][i] + 2*muh*previous_loss['flow_loss'][i]).to(device)

    if run != 0:
        run["aug_l/gen_ineq"].log(gen_ineq_loss_avg/nb_batches)
        run["aug_l/node_ineq"].log(node_ineq_loss_avg/nb_batches)
        run["aug_l/eq"].log(eq_loss_avg/nb_batches)
        run["aug_l/al_eq"].log(al_eq_loss_avg/nb_batches)
        #run["aug_l/al_flow"].log(al_flow_loss_avg/nb_batches)
        run["aug_l/flow"].log(flow_loss_avg/nb_batches)
        #run["train/true cost"].log(cost_loss_avg/nb_batches)
        run["train/avg_cost_difference"].log(cost_difference_avg/nb_batches)

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
    total_node_nb = characteristics['total_node_nb']
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
