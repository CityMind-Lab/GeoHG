from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.nn import GCNConv, to_hetero
from torch_geometric.nn import GraphConv
import torch
import copy

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
    
        self.conv1 = GraphConv((-1, -1), hidden_channels,aggr='mean')
        self.conv2 = GraphConv(hidden_channels, hidden_channels,aggr='mean')
        self.conv3 = GraphConv(hidden_channels, hidden_channels,aggr='mean')

        self.dropout = torch.nn.Dropout(p=0.5) 


    def forward(self, x_dict, edge_index, edge_attr):
        
        x_dict = self.conv1(x_dict, edge_index, edge_attr).relu()
     
        x_dict = self.conv2(x_dict, edge_index, edge_attr).relu()
        
        x_dict = self.dropout(x_dict)

        x_dict = self.conv3(x_dict, edge_index, edge_attr).relu()

        return x_dict
    

class MyModel(torch.nn.Module):
    def __init__(self, gnn, hidden_channels,out_channels,gnn_training=True):
        super().__init__()
        self.gnn = gnn
        
        for param in self.gnn.parameters():
            param.requires_grad = gnn_training
        self.dropout = torch.nn.Dropout(p=0.5)
        self.linear1 = torch.nn.Linear(hidden_channels,32)
        self.linear2 = torch.nn.Linear(32,16)
        self.linear3 = torch.nn.Linear(16,out_channels)


    def forward(self, data):
        x_dict = self.gnn(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
        
        x_dict = self.dropout(x_dict['area'])
        x_dict = self.linear1(x_dict).relu()
        x_dict = self.linear2(x_dict).relu()
        x_dict = self.dropout(x_dict)
        x_dict = self.linear3(x_dict)

        x_dict = {'area': x_dict}
        return x_dict