import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    TransformerConv,
    GCNConv,
    Linear
)

class NetGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()

        self.conv1 = GCNConv(in_channels = in_channels,_channels = hidden_channels, add_self_loops = True)
        self.bn1 = nn.BatchNorm1d(hidden_channels) #Seems to be a layer used to normalize each batch (has learnable parameters)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels)
        self.readout = nn.Sequential(
            # nn.Flatten(),
            #nn.Linear(hidden_channels, hidden_channels),
            #nn.Dropout(p=0.1),
            #nn.Tanh(), #Can be changed to sigmoid or others, but tanh is probably good enough
            #nn.Linear(hidden_channels, hidden_channels),
            #nn.Dropout(p=0.1),
            #nn.Tanh(),
            #nn.Linear(hidden_channels, hidden_channels),
            #nn.Dropout(p=0.1),
            #nn.Tanh(),
            nn.Linear(hidden_channels, 32),
            nn.Dropout(p=0.1),
            nn.Tanh(),
            nn.Linear(32, out_channels),
        )

    def bn(self, i, x):
        num_nodes, num_channels = x.size()

        x = x.view(-1, num_channels)
        x = getattr(self, f"bn{i}")(x)
        x = x.view(num_nodes, num_channels)
        return x

    def forward(self, x):
        x0 = x
        #print(f"before messge passing/{x0}")
        x1 = self.bn(1, self.conv1(x0.x, x0.edge_index)).tanh()  # emb dim: n_nodes * n_feat_node [2] = n_nodes * n_feat_node [hidden_channels]
        #print(f"after messge passing/{x1}")
        #x2 = self.bn(2, self.conv2(x1, x0.edge_index)).tanh()  # emb dim: n_nodes * n_feat_node [hidden_channels] = n_nodes * n_feat_node [hidden_channels]
        #x3 = self.bn(3, self.conv3(x2, x0.edge_index)).tanh()  # fin emb_dim: n_nodes * n_feat_node [hidden_channels]
        #mlp = self.readout(x2)
        mlp = self.readout(x1)
        #print(f"after processing/{mlp}")

        return mlp

mid = 24
#bot = 8

class NetGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        num_heads = 2
        #self.conv1 = nn.Linear(in_channels, hidden_channels)
        #self.conv1 = GCNConv(in_channels = in_channels, hidden_channels = hidden_channels, add_self_loops = True)
        self.conv1 = TransformerConv(in_channels = in_channels, out_channels = hidden_channels, heads=num_heads)
        self.bn1 = nn.BatchNorm1d(hidden_channels * num_heads)
        self.conv2 = TransformerConv(hidden_channels * num_heads, hidden_channels * num_heads)
        #self.conv2 = TransformerConv(hidden_channels * num_heads, out_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels * num_heads)
        self.conv3 = TransformerConv(hidden_channels * num_heads, hidden_channels * num_heads)
        self.bn3 = nn.BatchNorm1d(hidden_channels * num_heads)
        self.readout = nn.Sequential(

            nn.Linear(hidden_channels * num_heads, mid * num_heads),
            #nn.Linear(in_channels, mid * num_heads),
            #nn.Linear(hidden_channels * num_heads, bot),
            #nn.Dropout(p=0.1),
            #nn.Tanh(),
            nn.Tanhshrink(),

            nn.Linear(mid * num_heads,mid* num_heads),
            #nn.Dropout(p=0.1),
            nn.Tanhshrink(),

            #nn.Linear(mid * num_heads,mid* num_heads),
            #nn.Dropout(p=0.1),
            #nn.Tanhshrink(),

            #nn.Linear(mid * num_heads, bot),
            #nn.Dropout(p=0.1),
            #nn.Tanh(),
            #nn.Tanhshrink(),

            #print('out channels'),
            #print(out_channels),mid * num_heads
            #nn.Linear(hidden_channels * num_heads, out_channels),
            nn.Linear(mid * num_heads, out_channels),
            #nn.Linear(bot, out_channels),
            #nn.Tanhshrink()
            #nn.Softplus()
            #nn.Sigmoid()
        )

    def bn(self, i, x):
        num_nodes, num_channels = x.size()

        x = x.view(-1, num_channels)
        #x = getattr(self, f"bn{i}")(x)
        x = x.view(num_nodes, num_channels)
        return x

    def forward(self, x):
        x0 = x
        #print('Inside the algo')
        #print(x0.x)
        x1 = self.bn(1, self.conv1(x0.x, x0.edge_index)) # emb dim: n_nodes * n_feat_node [2] = n_nodes * n_feat_node [hidden_channels]
        #x1 = F.tanhshrink(x1)
        x2 = self.bn(2, self.conv2(x1, x0.edge_index))  # emb dim: n_nodes * n_feat_node [hidden_channels] = n_nodes * n_feat_node [hidden_channels]
        #x2 = F.tanhshrink(x2)
        x3 = self.bn(3, self.conv3(x2, x0.edge_index))  # fin emb_dim: n_nodes * n_feat_node [hidden_channels]
        #x4 = self.bn(4, self.conv4(x3, x0.edge_index))  # fin emb_dim: n_nodes * n_feat_node [hidden_channels]
        
        #return x2
        
        #mlp = self.readout(x1)
        
        mlp = self.readout(x3)
        return mlp

class NetGATmulti(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        num_heads = 4
        #self.conv1 = nn.Linear(in_channels, * hidden_channels)
        self.conv1 = TransformerConv(in_channels = in_channels, out_channels = hidden_channels, heads=num_heads)
        self.bn1 = nn.BatchNorm1d(hidden_channels * num_heads)
        self.conv2 = TransformerConv(hidden_channels * num_heads, hidden_channels * num_heads)
        #self.bn2 = torch.nn.BatchNorm1d(hidden_channels * num_heads)
        self.conv3 = TransformerConv(hidden_channels * num_heads, hidden_channels * num_heads)
        self.bn3 = nn.BatchNorm1d(hidden_channels * num_heads)
        self.readout = nn.Sequential(
            #nn.Linear(hidden_channels * num_heads, hidden_channels * num_heads),
            #nn.Tanh(),
            #nn.Linear(hidden_channels * num_heads, hidden_channels * num_heads),
            #nn.Dropout(p=0.1),
            #nn.Tanh(),
            ###
            #nn.Linear(hidden_channels * num_heads, hidden_channels * num_heads),
            #nn.Dropout(p=0.1),
            #nn.Tanh(),
            nn.Linear(hidden_channels * num_heads, 6 * num_heads),
            nn.Dropout(p=0.01),
            #nn.Tanh(),
            nn.Tanhshrink(),
            nn.Linear(6 * num_heads, 8),
            #nn.Dropout(p=0.1),
            #nn.Tanh(),
            nn.Tanhshrink(),
            #print('out channels'),
            #print(out_channels),
            nn.Linear(8, out_channels),
            #nn.Sigmoid()
        )

    def bn(self, i, x):
        num_nodes, num_channels = x.size()

        x = x.view(-1, num_channels)
        #x = getattr(self, f"bn{i}")(x)
        x = x.view(num_nodes, num_channels)
        return x

    def forward(self, x):
        x0 = x
        #print('Inside the algo')
        #print(x0.x)
        p = self.bn(1, self.conv1(x0.x, x0.edge_index)).tanh()  # emb dim: n_nodes * n_feat_node [2] = n_nodes * n_feat_node [hidden_channels]
        q = self.bn(1, self.conv1(x0.x, x0.edge_index)).tanh()  # emb dim: n_nodes * n_feat_node [2] = n_nodes * n_feat_node [hidden_channels]
        v = self.bn(1, self.conv1(x0.x, x0.edge_index)).tanh()  # emb dim: n_nodes * n_feat_node [2] = n_nodes * n_feat_node [hidden_channels]
        theta = self.bn(1, self.conv1(x0.x, x0.edge_index)).tanh()  # emb dim: n_nodes * n_feat_node [2] = n_nodes * n_feat_node [hidden_channels]

        #x2 = self.bn(2, self.conv2(x1, x0.edge_index)).tanh()  # emb dim: n_nodes * n_feat_node [hidden_channels] = n_nodes * n_feat_node [hidden_channels]
        #x3 = self.bn(3, self.conv3(x2, x0.edge_index)).tanh()  # fin emb_dim: n_nodes * n_feat_node [hidden_channels]
        mlp1 = self.readout(p)
        mlp2 = self.readout(q)
        mlp3 = self.readout(v)
        mlp4 = self.readout(theta)
        #mlp = torch.cat([mlp1,mlp2,mlp3,mlp4], dim= -1)
        mlp = 0
        return mlp
