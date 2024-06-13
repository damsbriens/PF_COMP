from GraphClassifGNN.Networks import *
from GraphClassifGNN import TrainValTest
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR, LinearLR
from torch import device, load
import torch
import torch_geometric


from torch.optim import Adam
#from PreProcess import build_OPFGraph
from torch_geometric import seed_everything
#print(torch_geometric.__version__)
import pickle
import argparse
#print(cuda.__version__)

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--euler", type=int)
args = parser.parse_args()
euler = args.euler          #euler == None means that it isn't euler and that we define the Neptune run


#from PreProcess_MG import return_func

# ----------------------------------------------------------------------------------------------------------------------------------
# User-defined inputs: 
type_v = "polar"  #' rect'
type_model = "AC_do"  # or'unsupervised'
eval_period = 10
max_epoch = 1500
#if euler == None:
#device = device('cpu')  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_neurons = 24
n_output = 4 

record = False

case = 'case14'
#case = "case24_ieee_rts"
#case = "case24_ieee_rts_modified"
#case = "case9Q"
#case = "case9_modified"
#case = 'case9Q_test_N-1'
#case = 'case30'
#case = 'case30Q'
#case = 'case118'
#case = 'case145'

points = 1
batch_size = 1
gamma0 = 0.9995 #0.9995
lr0= 0.00001
#lr0= 0.0001
epochs = 10000
penalty_multiplers = [5, 0.001, 0.1, 1.0001, 1.00002, 1.00003] #These multiplers are generally static except for mun0, which is a balancing term to decide of we want to focus more on cost or eq loss
training_params = [gamma0,lr0, case, epochs]
# ----------------------------------------------------------------------------------------------------------------------------------
 
#euler = 1

if euler == None:
    print('define the run')
    import neptune
    if record:
         run = neptune.init_run(
              #name = "Node_loss_test-1-AL",
              project="opfpinns/OpfPinns-MT-Experiments",
              api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzZTk0ZDZmNC1mZGNmLTRjODEtOGY3Mi0yMTViYzRjNzRiYmQifQ==",
              )
    else:
         run = neptune.init_run(  #Send the debugging stuff to a private workspace (Anna still has access)
              project="OpfPINNs2/OpfPINNS-MT",
              api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzZTk0ZDZmNC1mZGNmLTRjODEtOGY3Mi0yMTViYzRjNzRiYmQifQ==",
              )
else:
    print('dont define the run')
    run = 0

names = ["train", "val", "test"]

if run !=0:
    run["parameters/gamma0"] = gamma0
    run["parameters/lr0"] = lr0

 

if __name__ == "__main__":
     #"""
     #characteristics, train_loader, val_loader, test_loader = build_OPFGraph(case, points, spread = 0.0 ,split= [0.8, 0.1, 0.1], batch_size = batch_size)
     #print(characteristics)
     print('Loading {}_{}_{}.pt'.format(case, points, batch_size))

     train_loader = torch.load("Input_Data/train_loader_{}_{}_{}.pt".format(case, points, batch_size))
     val_loader = torch.load("Input_Data/val_loader_{}_{}_{}.pt".format(case, points, batch_size))
     test_loader = torch.load("Input_Data/test_loader_{}_{}_{}.pt".format(case, points, batch_size))
     with open('Input_Data/characteristics_{}.pkl'.format(case), 'rb') as f:
            characteristics = pickle.load(f)

     #Loads the data
     datasets = [
        train_loader,
        val_loader,
        test_loader,
     ]

     seed_everything(2000)
     model = NetGAT(train_loader.dataset[0].num_features, n_neurons, n_output).to(device)
     #case = 'case9Q'
     #model = torch.load("Results\Saved_Models\{}_2GAT\model_{}_2GAT.pt".format(case,case))     
     
     optimizer = Adam(model.parameters(), lr=lr0, weight_decay=0.0001)
     # model.load_state_dict(torch.load("C:\\Users\\avarbella\\Documents\\GraphGym-master\\"
     #                                 "GraphGym-master\\Cascades_GNN\\Training\\models\\model_swiss_bc_update.pt"))
     #REDUCELRONPLATEAU
     #scheduler = LinearLR(optimizer)
     scheduler = ExponentialLR(optimizer, verbose=False, gamma=gamma0
     )
     #scheduler = ReduceLROnPlateau(#ExponentialLR( #Scheduler is what allows to adapt the learning rate based on how well the model is learning
     #    optimizer, mode='min', factor=0.1, patience = 100, verbose=False# optimizer, verbose=True  #, gamma=0.8
     #)
     #scheduler = ExponentialLR( optimizer, verbose=True  , gamma=0.05)
     loaders = datasets
     #"""
     """
     TrainValTest.test_numerical_loss(
        graph,characteristics, model, device, run
     )
     #"""
     #"""
     TrainValTest.train_regr_pinn_correct(
         loaders,
         characteristics,
         model,
         optimizer,
         device=device,
         max_epoch=max_epoch,
         eval_period=eval_period,
         run=run,
         scheduler=scheduler,
         training_params = training_params,
         batch_size= batch_size,
         augmented_epochs= epochs,
         penalty_multipliers= penalty_multiplers
     )
     #"""