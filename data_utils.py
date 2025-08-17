import pandas as pd
import numpy as np
import torch
import random
from torch.utils.data import Dataset

def load_node(txt_path):
    node_map = {}
    node_type = None
    edges = []
    
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip().split()
        # print(line)
        if node_type==None:
            node_type = line[0].split('/')[0]
        node1 = int(line[0].split('/')[1])
        node2 = int(line[2].split('/')[1])
        if node1 not in node_map.keys():
            node_map[node1] = len(node_map) 
        
        if node2 not in node_map.keys():
            node_map[node2] = len(node_map) 
        
        edges.append((node_map[node1], node_map[node2]))
    
    return node_map, edges



def load_relation(txt_path,src_map,dst_map,thresh=0.0):
    edges = []
    edges_attr = []
    
    with open(txt_path, 'r') as f:
        lines = f.readlines()
   
    for line in lines:
        line = line.strip().split()
        src = int(line[0].split('/')[1])
        dst = int(line[2].split('/')[1])
        portion = float(line[3].split('/')[1])
        if portion>=thresh:
            edges_attr.append([portion])  
            edges.append((src_map[src],dst_map[dst]))

        
    return edges,edges_attr

def load_y(predit_file_path,area_node_map):
    y_dict = {}
    with open(predit_file_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip().split()
        txt_id = int(line[0].split('/')[1])
        this_y = float(line[1].split('/')[1])
       
        graph_id = area_node_map[txt_id]
        y_dict[graph_id] = this_y

  
    y = [y_dict[i] for i in range(len(y_dict))]
    return y

def load_pos_encode(pos_file_path,area_node_map):
    pos_dict = {}
    with open(pos_file_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip().split(',')
        area_txt_id = int(line[0])
        x = float(line[1])
        y = float(line[2])
        
        graph_id = area_node_map[area_txt_id]
        pos_dict[graph_id] = (x, y)
   
    pos_encode = [pos_dict[i] for i in range(len(pos_dict))]
    return pos_encode
        
def load_tif_feature(tif_file_path,area_node_map):
    tif_dict = {}
    df = pd.read_csv(tif_file_path)
   
    df = df.fillna(0)
    # Return all rows from the third column to the second to last column, one line per tuple; 
    # the first column is file_name, the second column is coordinate, ignore
    area_values = df.iloc[:, 2:-1].values
    for i in range(len(area_values)):
        graph_id = area_node_map[area_values[i][0]]
        tif_dict[graph_id] = area_values[i][1:]
    
    tif_feature = [tif_dict[i] for i in range(len(tif_dict))]
    return np.array(tif_feature)

def load_poi_feature(poi_file_path,area_node_map):
    poi_dict = {}
    df = pd.read_csv(poi_file_path,index_col=False)
    df = df.fillna(0)

    # Return all rows from the second column to the penultimate column, one row per tuple;
    area_values = df.iloc[:, :-1].values
    for i in range(len(area_values)):
        graph_id = area_node_map[area_values[i][0]]
        if graph_id not in poi_dict.keys():
            poi_dict[graph_id] = area_values[i][1:]
        else:
            print("Duplicate area node id, the area node id in txt is:",area_values[i][0])

    poi_feature = [poi_dict[i] for i in range(len(poi_dict))]
    return np.array(poi_feature)


def normalize_y(y,task):
    y = torch.tensor(y)
    if task=='Carbon':
        shift = 500
    
        y[y>2000] = 2000
    else:
        shift = 1


    y[y<0] = 0

    min_value = torch.min(y)
    shifted_values = y - min_value + shift

    log_transformed_values = torch.log(shifted_values)

    mean = torch.mean(log_transformed_values)
    std = torch.std(log_transformed_values)
    # print(mean,std)
    # normalization
    log_normalized_y = (log_transformed_values - mean) / std
    return log_normalized_y

def denormalize_y(log_normalized_y, task,mean,std):
    # mean = torch.mean(log_normalized_y)
    # std = torch.std(log_normalized_y)

    # Perform denormalization operation
    log_transformed_values = log_normalized_y * std + mean

    # de-log transformation
    shifted_values = torch.exp(log_transformed_values)

    if task == 'Carbon':
        shift = 500

        shifted_values[shifted_values > 2000] = 2000
    else:
        shift = 1

    y = shifted_values - shift


    return y

def train_val_test_split(data, train_ratio=0.6, val_ratio=0.2,test_ratio=0.2):
    num_nodes = data['area'].num_nodes
    data['area'].train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data['area'].val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data['area'].test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    random_area_id = torch.randperm(num_nodes)
    train_size = int(len(random_area_id)*train_ratio)
    val_size = int(len(random_area_id)*val_ratio)
    test_size = int(len(random_area_id)*test_ratio)
    data['area'].train_mask[random_area_id[:train_size]] = True
    data['area'].val_mask[random_area_id[train_size:train_size+val_size]] = True
    data['area'].test_mask[random_area_id[train_size+val_size:train_size+val_size+test_size]] = True
    return data


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def pairwise_cosine_similarity(tensor):
    """
    Compute pairwise cosine similarity of vectors in a tensor.
    
    Args:
        tensor (Tensor): Input tensor of shape (n, 128).
        
    Returns:
        Tensor: Pairwise cosine similarity matrix of shape (n, n).
    """
    # Compute L2 norms of vectors
    norms = torch.norm(tensor, p=2, dim=1, keepdim=True)
    # Normalize vectors
    normalized_tensor = tensor / norms
    # Compute pairwise cosine similarity
    similarity_matrix = torch.mm(normalized_tensor, normalized_tensor.t())
    return similarity_matrix

def save_result(args, best_epoch, best_score, savepath):
    data = {
    'Best Epoch': best_epoch,
    'Val ' +args.metrics: best_score,
    **vars(args)
    }

    # 将字典转换为DataFrame
    df = pd.DataFrame([data])
    # 写入CSV文件
    df.to_csv(savepath, index=False)

class SSLDataset(Dataset):
    def __init__(self, data,neigh_num):
        self.num_nodes = data['area'].num_nodes
        self.neigh_num = neigh_num

        self.x_entity_portion = data['area'].x[:, 2:11]
        # print(self.x_entity_portion[0])
        similarity_matrix = pairwise_cosine_similarity(self.x_entity_portion)
        # print(similarity_matrix.shape)
        self.similarity_matrix = similarity_matrix
    

        self.neighs = {}
        self.pos = {}

        edges = data['area', 'near', 'area'].edge_index.T
        for i in range(self.num_nodes):
            self.neighs[i] = []

            self.neighs[i] += list(edges[edges[:, 0] == i][:, 1])
            sim = self.similarity_matrix[i]
            top_values, top_indices = torch.topk(sim, k=1000)
            # print(top_values[:5])
            self.neighs[i] += list(top_indices)



            if i in self.neighs[i]:
                self.neighs[i].remove(i)

            self.neighs[i] = self.neighs[i][:self.neigh_num]
  
            self.neighs[i] = [int(x) for x in self.neighs[i]]


    def __len__(self):
        return self.num_nodes

    def __getitem__(self, idx): 

        sample = {
            "center_id": idx,
            "neighbors_id": np.array(self.neighs[idx]), # list

        }


        return sample 

