import torch
from torch import nn, optim
from torch.nn import functional as F

import argparse

from trainer import *
from dataloader import *
from data_utils import *
from eval_utils import *
from trainer import *
from cope import CoPE

import networkx as nx

# from teneto import TemporalNetwork

ending_time = 1.
burnin_time = 0.0
alpha = 0.98
hidden_size = 32
# hidden_size = 128
n_neg_samples = 16

tbptt_len = 20
delta_coef = 0.


def load(name):
    if name in {'wikipedia_test2','cyverse', 'wikipedia_test', 'wikipedia', 'lastfm', 'reddit'}:
        df, feats = load_jodie_data(f'data/{name}.csv')
    else:
        df, feats = load_recommendation_data(f'data/{name}_5.csv')
    return df, feats


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-3)
    # parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    args = parser.parse_args()
    device = torch.device(f'cuda:{args.cuda}')
    df, feats = load(args.dataset)
    n_users, n_items = df.iloc[:, :2].max() + 1

    print("n_users: ", n_users)
    print("n_items: ", n_items)
    print("df: ", df)

    G = nx.Graph()

    df_temp = df
    df_temp['item_id'] = df_temp['item_id'] + n_users

    G = nx.from_pandas_edgelist(df_temp, 'item_id', 'user_id')

    print("G: ", G)
    
    centrality = nx.eigenvector_centrality(G)
    centrality = np.fromiter(centrality.values(), dtype=float)


    # temp_df = df[['user_id', 'item_id', 'timestamp']]
    # temp_df.rename({'user_id': 'i', 'item_id': 'j', 'timestamp' : 't'}, axis=1, inplace=True)

    # tnet = TemporalNetwork(from_df = temp_df)

    # centrality = temporal_closeness_centrality(tnet = tnet)

    # attn = pd.DataFrame(centrality)

    print("centrality: ", centrality.max())
    print("centrality: ", len(centrality))
    print("centrality: ", type(centrality))

    n_nodes = n_users + n_items

    # attn = torch.ones(n_nodes) * 3

    attn = torch.from_numpy(centrality).float()

    
    print("attn2: ", attn)
    print("attn2: ", len(attn))
    print("attn2: ", type(attn))
