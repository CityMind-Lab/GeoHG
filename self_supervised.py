'''
    GeoHG-SSL: Self-supervised contrastive learning to obtain pre-trained graph embedding
'''
import argparse
import logging
import datetime
import time
import os
from torch_geometric.data import HeteroData
import torch
import torch_geometric.transforms as T
import random
import numpy as np
import copy
import torch_geometric
from torch.utils.data import DataLoader

from data_utils import *
from model import *

import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

def cal_cl_loss(s_emb, t_emb, labels):
    # Use a fixed temperature scale; avoid creating new parameters on every call
    logit_scale = torch.tensor(1 / 0.07, device=s_emb.device)
    logits = logit_scale * s_emb @ t_emb.t()
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    # print(loss_i,loss_t )
    ret_loss = (loss_i + loss_t) / 2 
    return ret_loss

def train(data,dataset,model,args):
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,weight_decay=2e-4)
    
    model.train()

    losses = []

    min_loss = float('inf')

    for epo in range(args.epoch_num):
        epoch_loss = 0.0
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        for batch in loader:
            # Move indices to the correct device for safe tensor indexing
            center_id = batch['center_id'].to(device)
            neighs_id = batch['neighbors_id'].to(device)
            
            x_dict = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
            
            neighs_emb = x_dict['area'][neighs_id]
          
            neighs_emb = torch.mean(neighs_emb,dim=1,keepdim=False)
      
            center_emb = x_dict['area'][center_id]
            
            optimizer.zero_grad()
            loss = cal_cl_loss(center_emb, neighs_emb, torch.arange(args.batch_size).to(device))
        
            loss.backward()
            optimizer.step()
            epoch_loss += loss
        epoch_loss = epoch_loss / len(loader)
        
        losses.append(epoch_loss.item())
        if epoch_loss < min_loss:
            min_loss = epoch_loss
            best_model = copy.deepcopy(model)
            torch.save(best_model.state_dict(), os.path.join(args.model_saved_dir, log_name.replace('.log','.pth')))
            flag = "*"
        else:
            flag = ""
        print(f'Epoch: {epo:03d}, Loss: {epoch_loss:.4f} ,Min Loss: {min_loss:.4f} {flag}')
        
        if min(losses[-20:]) > min_loss and args.early_stop:
            print('Early stopping')
            break
        
    return losses,best_model


def main(args):
    setup_seed(0)

    # ANA relation file path
    ana_file_path = os.path.join(args.graph_data_dir, args.city, 'ANA.txt')
    # ELA relation file path
    ela_file_path = os.path.join(args.graph_data_dir, args.city, 'ELA.txt')
    # PLA relation file path
    pla_file_path = os.path.join(args.graph_data_dir, args.city, 'PLA.txt')
    # area pos embedding 文件路径
    pos_file_path = os.path.join(args.graph_data_dir, args.city, 'pos_encode.txt')
    # geo-entity file path
    tif_file_path = os.path.join(args.graph_data_dir, args.city, 'TIF_feature.csv')
    # poi information file path
    poi_file_path = os.path.join(args.graph_data_dir, args.city, 'POI_feature.csv')


    logging.info("Start loading data...")
    
    area_node_map, ana_edges = load_node(ana_file_path) # area_node_map: {area_in_txt_id: area_in_graph_id}
    logging.info("area node num: {}".format(len(area_node_map)))
    logging.info("area near area edge num: {}".format(len(ana_edges)))

    entity_node_map = {i:i-1 for i in range(1,10)}  
    poi_node_map = {i:i for i in range(0,14)}

    
    ela_edges, ela_edges_attr = load_relation(ela_file_path,entity_node_map,area_node_map,args.entity_thresh)
    logging.info("entity locate area edge num: {}".format(len(ela_edges)))

   
    pla_edges, pla_edges_attr = load_relation(pla_file_path,poi_node_map,area_node_map,args.poi_thresh)

  
    pos_encode = load_pos_encode(pos_file_path,area_node_map)

    
    tif_feature = load_tif_feature(tif_file_path,area_node_map)

  
    poi_feature = load_poi_feature(poi_file_path,area_node_map)

 
    data = HeteroData()
    data['area'].num_nodes = len(area_node_map)

 
    # Build node features robustly even when position embedding is disabled
    feature_tensors = []
    if args.pos_embedding:
        pos_tensor = torch.tensor(pos_encode, dtype=torch.float32)
        # Standardize the two positional dimensions
        pos_tensor[:, 0] = (pos_tensor[:, 0] - torch.mean(pos_tensor[:, 0])) / torch.std(pos_tensor[:, 0])
        pos_tensor[:, 1] = (pos_tensor[:, 1] - torch.mean(pos_tensor[:, 1])) / torch.std(pos_tensor[:, 1])
        feature_tensors.append(pos_tensor)

    if args.hypernode in ['all', 'entity']:
        feature_tensors.append(torch.tensor(tif_feature, dtype=torch.float32))
    if args.hypernode in ['all', 'poi']:
        feature_tensors.append(torch.tensor(poi_feature, dtype=torch.float32))

    if len(feature_tensors) > 0:
        data['area'].x = torch.cat(feature_tensors, dim=1)
    else:
        # No features selected; keep a zero-width feature matrix
        data['area'].x = torch.zeros((len(area_node_map), 0), dtype=torch.float32)
    
    logging.info(f"X feature shape:{data['area'].x.shape}")

    data['area'].x = data['area'].x.float()

    if args.hypernode =='all' or args.hypernode == 'entity':
         
            data['entity'].num_nodes = len(entity_node_map)
            data['entity'].x = torch.zeros(9, 9)
            for n in range(9):
                data['entity'].x[n, n] = 1
            data['entity', 'locate', 'area'].edge_index = torch.tensor(ela_edges, dtype=torch.long).t().contiguous() # shape: [2, num_edges=24902]
            data['entity', 'locate', 'area'].edge_attr = torch.tensor(ela_edges_attr, dtype=torch.float).contiguous() # shape: [num_edges=24902, 1]
        
    if args.hypernode =='all' or args.hypernode == 'poi':
    
        data['poi'].num_nodes = len(poi_node_map)
        data['poi'].x = torch.zeros(14,14)
        for n in range(14):
            data['poi'].x[n, n] = 1
        data['poi', 'locate', 'area'].edge_index = torch.tensor(pla_edges, dtype=torch.long).t().contiguous() # shape: [2, num_edges=9957]
        data['poi', 'locate', 'area'].edge_attr = torch.tensor(pla_edges_attr, dtype=torch.float).contiguous() # shape: [num_edges=9957, 1]


    data['area','near','area'].edge_index = torch.tensor(ana_edges, dtype=torch.long).t().contiguous()  # shape: [2, num_edges=67172]

   
    data = T.ToUndirected()(data)
    logging.info(f"Graph data:\n{str(data)}")

 
    logging.info("Start building dataset...")
    data = data.to('cpu')
    dataset = SSLDataset(data, neigh_num=args.neigh_num)
    data = data.to(device)
    
  
    gnn = GNN(args.hidden_channels, out_channels=1).to(device)
    gnn = to_hetero(gnn, (data.node_types, data.edge_types), aggr='mean').to(device)

    logging.info("Start training...")
    losses,best_model = train(data,dataset,gnn,args)

 
    import matplotlib.pyplot as plt
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss of Self-supervised learning')
    plt.savefig(os.path.join(args.log_saved_dir, log_name.replace('.log','.png')))
    plt.show()

  






if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--city", type=str, default="GZ", choices=['GZ','SZ','BJ','SH'],help="city abbreviation")
    parser.add_argument("--pos_embedding",type=bool,default=True,help='Whether to use position embedding in node features')
    parser.add_argument("--hypernode", type=str, default="all", choices=['all','entity','poi','mono'],help="use what kinds of nodes as hypernode")
    parser.add_argument("--entity_thresh",type=float,default=0.0,help='filt out entity hyperlinks whose weight is lower than threshold.')
    parser.add_argument("--poi_thresh",type=float,default=0.0,help='filt out poi hyperlinks whose weight is lower than threshold.')
    
    parser.add_argument("--graph_data_dir", type=str, default="data/Hyper_Graph", help="graph data directory")
    parser.add_argument("--model_saved_dir", type=str, default="data/models", help="model save directory")
    parser.add_argument("--log_saved_dir", type=str, default="data/logs", help="log save directory")
    parser.add_argument("--neigh_num", type=int, default=5, help="similar neighbors num")
    parser.add_argument("--epoch_num", type=int, default=400, help="epoch number")
    parser.add_argument("--early_stop", type=bool, default=True, help="early stop")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--random_seed", type=int, default=0, help="random seed")
    parser.add_argument("--gpu", type=int, default=0)

    # gnn config
    parser.add_argument("--hidden_channels", type=int, default=64)

    args = parser.parse_args()

    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")

    # logger
    log_name = 'ssl_{}_{}.log'.format(args.city,datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=args.log_saved_dir + '/' + log_name,
                        filemode='w')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)  # 控制台输出的日志级别

    # 创建一个格式器，定义输出格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 将格式器添加到控制台处理器
    console.setFormatter(formatter)

    # 将控制台处理器添加到根记录器
    logging.getLogger('').addHandler(console)

    # 输出args
    logging.info(args)

    start = time.perf_counter()
    main(args)
    end = time.perf_counter()
    logging.info("Total time: {}".format(end - start))

    
   