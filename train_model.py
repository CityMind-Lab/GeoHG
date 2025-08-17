import argparse
import logging
import datetime
import time
import os
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
import random
import numpy as np
from torch_geometric.loader import NeighborLoader
import networkx as nx
import torch_geometric
from torch.utils.data import DataLoader
from data_utils import *
import matplotlib.pyplot as plt
from torch_geometric.transforms import RandomNodeSplit
from model import *


import torch.nn.functional as F
from tqdm import tqdm

@torch.no_grad()
def test(data, model, split='val', metrics='rmse'):
    data.to(device)
    model.eval()

    out = model(data)
    y = data['area'].y
    mask = getattr(data['area'], f'{split}_mask')
    rmse = F.mse_loss(out['area'][mask].squeeze(), y[mask]).sqrt()
    r2 = 1 - F.mse_loss(out['area'][mask].squeeze(), y[mask]) / F.mse_loss(y[mask].mean(), y[mask])
    mae = F.l1_loss(out['area'][mask].squeeze(), y[mask])

    metrics_dict = {'rmse': rmse, 'r2': r2, 'mae': mae}

    if split=='test':
        df = pd.read_csv(os.path.join(args.log_saved_dir,log_name.replace('.log','_res.csv')))

        df.insert(0,'rmse',rmse.item())
        df.insert(0,'r2',r2.item())
        df.insert(0,'mae',mae.item())
        df.to_csv(os.path.join(args.log_saved_dir,log_name.replace('.log','_res.csv')), index=False)

    return metrics_dict[metrics]
    
    

def train(data,model,args):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    data.to('cuda')
    model.train()

    losses = []
    train_metrics = []
    val_metrics = []
    if args.metrics == 'rmse':
        best_val_metirc = float('inf')
    elif args.metrics == 'r2':
        best_val_metirc = -float('inf')
        
    for epo in range(args.epoch_num):
        total_examples = torch.sum(data['area'].train_mask)
        total_loss = 0.0
        train_node_id = torch.where(data['area'].train_mask)[0]

        train_node_id = train_node_id[torch.randperm(len(train_node_id))]
        for i in range(0,total_examples,args.batch_size):
            optimizer.zero_grad()
            out = model(data)
            loss = F.mse_loss(out['area'][train_node_id[i:i+args.batch_size]].squeeze(), data['area'].y[train_node_id[i:i+args.batch_size]])
            # print(loss)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        loss = total_loss/total_examples

        train_metirc = test(data,model,'train',args.metrics)
        val_metric = test(data,model,'val',args.metrics)

        losses.append(loss.item())
        train_metrics.append(train_metirc.item())
        val_metrics.append(val_metric.item())

        if args.metrics == 'rmse' or args.metrics == 'mae':
            if val_metric < best_val_metirc:
                best_val_metirc = val_metric
                best_epoch = epo
                best_model = copy.deepcopy(model)
                flag = "*"
            else:
                flag = ""
        elif args.metrics == 'r2':
            if val_metric > best_val_metirc:
                best_val_metirc = val_metric
                best_epoch = epo
                best_model = copy.deepcopy(model)
                flag = "*"
            else:
                flag = ""
        
        if flag=="*" and args.model_saved_dir!='':
            torch.save(model.state_dict(), os.path.join(args.model_saved_dir,log_name.replace('.log','_best_model.pth')))

        
        if args.early_stop:
            if args.metrics == 'rmse' and min(val_metrics[-100:])> best_val_metirc:
                logging.info(f"Early stop at epoch {epo}")
                break
            elif args.metrics == 'r2' and max(val_metrics[-100:]) < best_val_metirc:
                logging.info(f"Early stop at epoch {epo}")
                break
        
        if epo % 10 == 0:
            logging.info(f"Epoch: {epo}, Loss: {loss:.8f}, Train {args.metrics}: {train_metirc:.4f}, Val {args.metrics}: {val_metric:.4f} {flag}")

    save_result(args,best_epoch,best_val_metirc.item(),os.path.join(args.log_saved_dir,log_name.replace('.log','_res.csv')))
    return losses,train_metrics,val_metrics,best_model

def main(args):

    setup_seed(args.random_seed)
    # ANA relation file path
    ana_file_path = os.path.join(args.graph_data_dir, args.city, 'ANA.txt')
    # ELA relation file path
    ela_file_path = os.path.join(args.graph_data_dir, args.city, 'ELA.txt')
    # PLA relation file path
    pla_file_path = os.path.join(args.graph_data_dir, args.city, 'PLA.txt')
    # Y value file of the downstream task to be predicted
    predit_file_path = os.path.join(args.downstream_data_dir, args.city, f'{args.task}.txt')
    # area pos embedding file path
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

   
    y = load_y(predit_file_path,area_node_map)

    
    pos_encode = load_pos_encode(pos_file_path,area_node_map)

  
    tif_feature = load_tif_feature(tif_file_path,area_node_map)

   
    poi_feature = load_poi_feature(poi_file_path,area_node_map)

    data = HeteroData()
    data['area'].num_nodes = len(area_node_map)



    # Build node features robustly even when position embedding is disabled
    feature_tensors = []
    if args.pos_embedding:
        pos_tensor = torch.tensor(pos_encode, dtype=torch.float32)
        # normalization
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
        data['area'].x = torch.zeros((len(area_node_map), 0), dtype=torch.float32)
    
    logging.info(f"X feature shape:{data['area'].x.shape}")

    data['area'].x = data['area'].x.float()
    print(data['area'].x.shape)

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

    
    data["area"].y = normalize_y(y,args.task)


    data['area','near','area'].edge_index = torch.tensor(ana_edges, dtype=torch.long).t().contiguous()  # shape: [2, num_edges=67172]

   
    data = T.ToUndirected()(data)
    logging.info(f"Graph data:\n{str(data)}")

    
    data = train_val_test_split(data,train_ratio=0.7*(1-args.masked_ratio),val_ratio=0.3*(1-args.masked_ratio),test_ratio=args.masked_ratio)
    
    logging.info(f"train size: {data['area'].train_mask.sum().item()}, val size: {data['area'].val_mask.sum().item()}, test size: {data['area'].test_mask.sum().item()}")

    
    gnn = GNN(args.hidden_channels, out_channels=1)
    gnn = to_hetero(gnn, (data.node_types, data.edge_types), aggr='mean').to(device)
    if args.load_gnn!='':
        gnn.load_state_dict(torch.load(args.load_gnn))
    model = MyModel(gnn, args.hidden_channels, out_channels=1,gnn_training=args.gnn_training).to(device)

    logging.info("Start training...")
    losses,train_metrics,val_metrics,best_model = train(data,model,args)

    # training process visulization

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(losses, label='loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_metrics, label=f'train {args.metrics}')
    plt.plot(val_metrics, label=f'val {args.metrics}')
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.legend()

    plt.savefig(os.path.join(args.log_saved_dir, log_name.replace('.log','.png')))
    plt.show()

    logging.info("Start testing...")
    test_metric = test(data,best_model,'test',args.metrics)
    logging.info(f"Test {args.metrics}: {test_metric:.4f}")


    # visualization predicted values and ground true values
    with torch.no_grad():
        data = data.to('cuda')
        out = best_model(data)  # Move data to GPU

    real_values = data['area'].y[data['area'].test_mask].cpu()
    predicted_values = out['area'][data['area'].test_mask].squeeze().cpu()


    plt.figure(figsize=(18, 6))
    plt.plot(real_values.numpy(), label='Real Values', alpha=0.5, linewidth=0.8)
    plt.plot(predicted_values.numpy(), label='Predicted Values', alpha=0.5, linewidth=0.8)
    plt.xlabel('Node Index')
    plt.ylabel('Value')
    plt.title('Real Values vs Predicted Values')
    plt.legend()
    plt.savefig(os.path.join(args.log_saved_dir,log_name.replace('.log','_real_vs_pred.png')))
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--city", type=str, default="SZ", choices=['GZ','SZ','BJ','SH'],help="city abbreviation")
    parser.add_argument("--task", type=str, default="Population",choices=['Carbon','Light','Population','GDP','PM25'], help="task name")
    parser.add_argument("--pos_embedding",type=int,default=1,help='Whether to use position embedding in node features')
    parser.add_argument("--hypernode", type=str, default="all", choices=['all','entity','poi','mono'],help="use what kinds of nodes as hypernode")
    parser.add_argument("--entity_thresh",type=float,default=0.0,help='filt out entity hyperlinks whose weight is lower than threshold.')
    parser.add_argument("--poi_thresh",type=float,default=0.0,help='filt out poi hyperlinks whose weight is lower than threshold.')
    parser.add_argument("--graph_data_dir", type=str, default="data/Hyper_Graph", help="graph data directory")
    parser.add_argument("--downstream_data_dir", type=str, default="data/downstream_tasks", help="downstream data directory")
    parser.add_argument("--model_saved_dir", type=str, default="data/models", help="model save directory")
    parser.add_argument("--log_saved_dir", type=str, default="data/logs", help="log save directory")
    parser.add_argument("--epoch_num", type=int, default=2000, help="epoch number")
    parser.add_argument("--early_stop", type=bool, default=True, help="early stop")
    parser.add_argument("--batch_size", type=int, default=1000000)
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--random_seed", type=int, default=0, help="random seed")
    parser.add_argument("--masked_ratio", type=float, default=0.7, help="randomly masked data ratio")
    parser.add_argument("--metrics", type=str, default='r2', choices=['r2','rmse','mae'],help="evaluation metrics")
    parser.add_argument("--gpu", type=int, default=0)

    # gnn config
    parser.add_argument("--load_gnn", type=str, default="", help="pretrained gnn model path")
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--gnn_training", type=bool, default=True, help="whether to train gnn embedding model")

    args = parser.parse_args()

    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")

    # logger
    log_name = '{}_{}_{}.log'.format(args.city, args.task, datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=args.log_saved_dir + '/' + log_name,
                        filemode='w')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)  

    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    logging.info(args)

    if args.load_gnn!='':
        logging.info(f"Load pretrained GNN model from {args.load_gnn}")
    else:
        logging.info("Train GNN model from scratch")

    start = time.perf_counter()
    main(args)
    end = time.perf_counter()
    logging.info("Total time: {}".format(end - start))

    
   