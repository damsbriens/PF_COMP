import logging
import torch.nn as nn
#import torch
from utils import *
#import scipy.io
from training_steps import *
import pandas as pand
from time import time
from datetime import timedelta
from torch.profiler import profile, record_function, ProfilerActivity

PROFILE = False

def is_eval_epoch(cur_epoch, eval_period, max_epoch):
    """Determines if the model should be evaluated at the current epoch."""
    return (
        (cur_epoch + 1) % eval_period == 0
        or cur_epoch == 0
        or (cur_epoch + 1) == max_epoch
    )


#@torch.enable_grad
def train_regr_pinn_correct(loaders,characteristics, model, optimizer, device, max_epoch, eval_period, scheduler, run, training_params, batch_size, augmented_epochs, penalty_multipliers):
    '''
    This is one of the most important functions of all of the project. It defines the training iterations for both the penalty method and the augmented Lagrangian method.    
    '''

    previous_loss_dict = get_zero_constraints(model, loaders[0], characteristics, device, batch_size) #This is very important to start the AL method. Since the AL multipliers update with loss we need initial loss of 0 if we only do AL method



    prof = profile(
        activities=[ProfilerActivity.CPU],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=10, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("logs"),
        record_shapes=True,
    )



    # Training with the Augmented Lagrangian loss -------------------------------------------------------------------------------------------------------------------------
    #"""
    if PROFILE:
        prof.start()
    print('Training with augmented Lagrangian loss:')
    mun0, muf0, muh0, betan, betaf, betah  = [5, 0.001, 0.1, 1.0000, 1.00002, 1.00003]
    penalty_multipliers = penalty_multipliers_init(mun0, muf0, muh0, betan, betaf, betah)
    
    if run != 0: #Log these parameters into Neptune
        run["parameters/Mun0"] = mun0
        run["parameters/Muf0"] = muf0 
        run["parameters/Muh0"] = muh0
        run["parameters/Betan"] = betan
        run["parameters/Betaf"] = betaf
        run["parameters/Betah"] = betah
        run["parameters/AL epochs"] = augmented_epochs
    else:
        results_saver = np.zeros((augmented_epochs,2))
        val_loss_saver = []

    lambdas = init_lambda(loaders[0],characteristics, device, batch_size) # Initiates a vector full of 0 of the appropriate sizes for the AL multiplers
    t0 = time()
    for epoch in range(augmented_epochs):
        if run != 0 and is_eval_epoch(epoch, 10, max_epoch):
            print(" ################################ Epoch {} ################################".format(epoch))
        penalty_multipliers, previous_loss_dict, stats = training_step_coo_lagr(loaders[0], characteristics,model, optimizer, previous_loss_dict, lambdas, penalty_multipliers, run, device)
        #if run == 0:
        #    results_saver[epoch] += stats
        if is_eval_epoch(epoch, eval_period, max_epoch):
            #val_loss = eval_epoch_regr(loaders[1], model, device, run, characteristics)
            val_loss = 0
            if run == 0:
                val_loss_saver.append(val_loss)   
            #if epoch > 6000:
            scheduler.step()
            #scheduler.step(val_loss) #This is for the LR on Plateau scheduler
            #scheduler.step(previous_loss_dict['plate_loss'].sum())

        if PROFILE:
            prof.step()

    if PROFILE:
        prof.stop()
        print(prof.key_averages())

    #"""
    t1 = time()
    t = str(timedelta(seconds = t1-t0)) 
    print('function inferrence takes: ' + t)
    if run == 0:
        results = pand.DataFrame(results_saver, columns = ['Eq Loss', 'Cost'])
        pand.DataFrame.to_csv(results, 'Results/latest_results.csv')
        gamma0, lr0, case, epochs = training_params
        hyperparameters = pand.DataFrame(data = [mun0, muf0, muh0, betan, betaf, betah, augmented_epochs, gamma0, lr0, case, epochs], index = ['mun0', 'muf0', 'muh0', 'betan', 'betaf', 'betah', 'epochs', 'gamma0', 'lr0', 'case', 'epochs'])
        pand.DataFrame.to_csv(hyperparameters, 'Results/hyperparameters.csv')
        val_loss_saver = pand.DataFrame(data = np.array(val_loss_saver), columns = ['val_loss'])
        pand.DataFrame.to_csv(val_loss_saver, 'Results/val_loss.csv')

    test_epoch_regr_pinn(loaders[2], model, device, characteristics, run)

#@torch.no_grad()
def eval_epoch_regr(loader, model, device, run, characteristics): 
    '''
    Function used to evaluate the validation loss. Kind of obsolete since we don't really use validation loss as a real evaluation metric
    '''
    #print("Eval")
    model.eval()
    norm_coeffs = characteristics['norm_coeffs']
    edge_from_bus = characteristics["edge_to_bus"][0][0]
    edge_to_bus = characteristics["edge_to_bus"][0][1]
    gen_to_bus = characteristics["gen_to_bus"]
    edge_nb = len(characteristics['actual_edges'])
    base_MVA = characteristics['norm_coeffs']['Base_MVA']
    nb_nodes = characteristics['node_nb']
    edge_limits = characteristics['edge_limits']
    G = torch.tensor(characteristics['G']).unsqueeze(1)
    B = torch.tensor(characteristics['B']).unsqueeze(1)
    S_P = characteristics['S_P']
    SG_Q = characteristics['SG_Q']
    NG_V = characteristics['NG_V']
    v_target = []
    v_pred = []
    running_loss = list([0])
    eq_loss_avg = 0
    for n, data in enumerate(loader):
        data.to(device)
        batch_size = data.num_graphs
        true = data.y
        pred = model(data)
        
        # Denormalize the Outputs to properly compute the losses
        p_in = data.x[:, 0]*norm_coeffs['P_norm']
        q_in = data.x[:, 1]*norm_coeffs['Q_norm']
        V_in = data.x[:, 2]*norm_coeffs['V_norm']
        #print(p_in)

        V_in = V_in.view(batch_size, nb_nodes).T[:nb_nodes].to(device)
        p_in = p_in.view(batch_size, nb_nodes).T.to(device)
        q_in = q_in.view(batch_size, nb_nodes).T.to(device)

        p_out = pred[:, 0]*norm_coeffs['P_norm']
        q_out = pred[:, 1]*norm_coeffs['Q_norm']
        V_out = pred[:,2]*norm_coeffs['V_norm']
        Theta = pred[:,3]*(np.pi/180)*norm_coeffs['Theta_norm']


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
        #print(pd)
        eq_loss , flow_loss  = compute_eq_loss(p_in,q_in, V, p_out,q_out, Theta, edge_from_bus, edge_to_bus, gen_to_bus, base_MVA, G, B, batch_size, device)
        #print(eq_loss)
        #Pg = (gen_to_bus @ pg).to(device)
        #plate_loss = torch.abs(pd.sum() - Pg.sum())

        #plate_loss = torch.abs(torch.sum(pd) - pg[0] - pg[1] - pg[2])
        eq_loss_avg += torch.abs(eq_loss).sum()
        #plate_loss_avg += plate_loss

        #loss = nn.MSELoss()
        #loss = loss(pred, true)
        #running_loss[0] += loss.item()
        
        #v_target.append(true.detach())#.cpu())
        #v_pred.append(pred.detach())#.cpu())
    #if run != 0:
        #print("Val loss:", running_loss[0] / len(loader))
    #v_target = np.concatenate(v_target, axis=0)
    #v_pred = np.concatenate(v_pred, axis=0)
    #print(eq_loss_avg)    
    if run != 0:
        #run["val/batch/loss"].log(running_loss[0] / len(loader))
        run["val Eq loss"].log(eq_loss_avg/len(loader))
        #run["val plate loss"].log(plate_loss_avg/len(loader))
    return running_loss[0] / len(loader)

#@torch.no_grad()
def test_epoch_regr_pinn(loader, model, device, characteristics, run): #
    '''
    This is the final testing function once the model is trained. We look at equality losses and costs as the main evaluation metrics.
    I tend to print way too much stuff but it's never a bad thing to have more info.
    '''
    norm_coeffs = characteristics['norm_coeffs']
    model.eval()
    eq_losses = []
    true_eq_losses = []
    cost_differences = []
    trues = []
    preds = []
    nb_nodes = characteristics['node_nb']

    edge_from_bus = characteristics["edge_to_bus"][0][0]
    edge_to_bus = characteristics["edge_to_bus"][0][1]
    gen_to_bus = characteristics["gen_to_bus"]
    edge_nb = len(characteristics['actual_edges'])
    base_MVA = characteristics['norm_coeffs']['Base_MVA']
    gen_index = torch.tensor(characteristics['gen_index'])
    G = torch.tensor(characteristics['G']).unsqueeze(1)
    B = torch.tensor(characteristics['B']).unsqueeze(1)
    S_P = characteristics['S_P']
    SG_Q = characteristics['SG_Q']
    NG_V = characteristics['NG_V']

    i = 0
    running_loss = list([0])
    for data in loader:
        i += 1
        data.to(device)
        batch_size = data.num_graphs
        #true = data.y.detach()#.cpu()

        #print('True: ')
        #print(true)
        pred = model(data)
        #if i == 1:
        # Denormalize the Outputs to properly compute the losses
        p_in = data.x[:, 0]*norm_coeffs['P_norm']
        q_in = data.x[:, 1]*norm_coeffs['Q_norm']
        V_in = data.x[:, 2]*norm_coeffs['V_norm']
        print(p_in)
        print(q_in)
        print(V_in)

        V_in = V_in.view(batch_size, nb_nodes).T[:nb_nodes].to(device)
        p_in = p_in.view(batch_size, nb_nodes).T.to(device)
        q_in = q_in.view(batch_size, nb_nodes).T.to(device)

        p_out = pred[:, 0]*norm_coeffs['P_norm']
        q_out = pred[:, 1]*norm_coeffs['Q_norm']
        V_out = pred[:,2]*norm_coeffs['V_norm']
        Theta = pred[:,3]*(np.pi/180)*norm_coeffs['Theta_norm']


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

        print('Prediction: ')
        #print(pred)
        print(p_out.T)
        print(q_out.T)
        print(V_out.T)
        print(Theta.T)

        #loss = nn.MSELoss()
        #loss = loss(pred, true)
        #running_loss[0] += loss.item()
        
        #print(pg)
        #print(pd)
        #print(qg)
        #print(V)
        #print('Theta')
        #print(Theta)
        
        eq_loss = compute_eq_loss(p_in,q_in, V, p_out,q_out, Theta, edge_from_bus, edge_to_bus, gen_to_bus, base_MVA, G, B, batch_size, device) #We probably want to change this to get a better spread
        #plate_loss = torch.abs(torch.sum(pd) - pg[0] - pg[1] - pg[2])
        #print(true.T[0]*norm_coeffs['P_norm']) #true active power in MW

        #print(true_cost)
        
        eq_losses.append(abs(eq_loss).detach().sum())
        

        #trues.append(pand.DataFrame(true.detach().numpy()))
        preds.append(pand.DataFrame(pred.detach().numpy()))

        #print('Test part')
        """
        #eq_loss_true_2 , flow_loss_true, plate_loss_true  = compute_eq_loss_list(true.T[0]*norm_coeffs['P_norm'], pd, true.T[1]*norm_coeffs['Q_norm'], qd, true.T[2]*norm_coeffs['V_norm'], true.T[3]*np.pi/180*norm_coeffs['Theta_norm'], characteristics, batch_size, device)
        true_pg = true.T[0]*norm_coeffs['P_norm']
        true_qg = true.T[1]*norm_coeffs['Q_norm']
        true_V = true.T[2]*norm_coeffs['V_norm']
        true_Theta = true.T[3]*np.pi/180*norm_coeffs['Theta_norm']
        
        true_Theta = true_Theta.view(batch_size, nb_nodes).T[:nb_nodes]
        true_V = true_V.view(batch_size, nb_nodes).T[:nb_nodes]
        true_pg = true_pg.view(batch_size, nb_nodes).T
        true_qg = true_qg.view(batch_size, nb_nodes).T
        true_Theta[characteristics['Ref_node'],:] = 0
        static_costs = torch.full((1,batch_size), characteristics['static_costs'].sum()).squeeze(0)
        #print(true_pg)
        #print(pg)
        
        eq_loss_true , flow_loss_true = compute_eq_loss(p_in,q_in, V, p_out,q_out, Theta, edge_from_bus, edge_to_bus, gen_to_bus, base_MVA, G, B, batch_size, device)

        #print(eq_loss_true)
        #print(eq_loss_true_2)

        #print(eq_loss_true)
        print('True Eq loss: ' + str(abs(eq_loss_true).detach().sum()))
        #print(abs(eq_loss_true_2).detach().sum())
        #print('True Flow loss: ' + str(abs(flow_loss_true).detach().sum()))
        true_eq_losses.append(abs(eq_loss_true).detach().sum())
        """
    
    avg_true_eq_loss = np.mean(np.array(true_eq_losses))
    avg_eq_loss = np.mean(np.array(eq_losses))
    avg_cost_difference = np.mean(np.array(cost_differences))
    print('Average equality loss: ' + str(avg_eq_loss))
    print('Average cost difference: ' + str(avg_cost_difference))
    #print('Transmission limit violations: ' + str(flow_loss))
    #print('Average true equality loss: ' + str(avg_true_eq_loss))
    #print(flow_loss_true)

    #print(eq_losses)

    #trues = pand.concat(trues)
    preds = pand.concat(preds)

    print(eq_losses)
    print(cost_differences)

    print('Average MSE Loss: ' + str(running_loss[0] / len(loader)))

    #relevant_results_saver = pand.DataFrame(data = np.array([eq_losses, cost_differences]), index = ['eq_losses', 'cost_differences'])
    #pand.DataFrame.to_csv(relevant_results_saver, 'Results/relevant_results.csv')

    #pand.DataFrame.to_csv(trues, 'Results/true.csv')
    pand.DataFrame.to_csv(preds, 'Results/pred.csv')

    torch.save(model, 'Results/model.pt')
