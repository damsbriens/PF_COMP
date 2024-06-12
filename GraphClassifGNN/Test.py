import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
)
import torch.nn.functional as F
from torch_geometric.nn import (
    global_max_pool,
    TransformerConv,
    global_mean_pool,
    global_add_pool,
)
import torch
import torch.nn as nn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



def predict_regr(loader, model, device):
    model.eval()
    v_pred = []
    v_target = []
    for data in loader:
        #data.to(torch.device(device))
        preds = model(data)
        v_pred.append(preds.float().detach().cpu())
        v_target.append(data.y.detach().cpu())
    v_pred = np.concatenate(v_pred, axis=0)
    v_target = np.concatenate(v_target, axis=0)
    return v_pred, v_target


def predict_r(loader, model):
    #model.eval()
    v_pred = [model(data).detach().cpu().numpy() for data in loader]
    v_pred = np.concatenate(v_pred, axis=0)

    return v_pred


