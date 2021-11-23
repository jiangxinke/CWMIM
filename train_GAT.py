# matplotlib.use('Agg')
import argparse
import os
import torch
import numpy as np
import torch.optim as optim
import lib.utils as utils
from lib.args import add_args
from torch.utils.data import DataLoader, Subset
from lib.dataloader import Slope
from lib.layers.GNN import GCN, VGAE, adj_from_mean
from lib.layers.GAT import GATNet
from Trainers import StdTrainer, GATTrainer

# Load and initialize other parameters
parser = argparse.ArgumentParser('StdModel')
add_args(parser)
args = parser.parse_args()
args.device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
args.save_dir = os.path.join('results', args.data, args.log_key)
args.fig_save_dir = os.path.join(args.save_dir, 'figs')
args.log_dir = os.path.join(args.save_dir, 'logs')
utils.makedirs(args.save_dir)
utils.makedirs(args.fig_save_dir)
# utils.makedirs(args.log_dir)
utils.set_random_seed(args.seed)


if __name__ == '__main__':
    # load data
    slope_set = Slope(args.data_path, seq_len=args.seq_len, pre_len=args.pre_len)
    scaler = slope_set.scaler
    train_set = Subset(slope_set, range(0, int(slope_set.len * args.train)))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    # initialize
    model = GATNet(in_c=args.seq_len, hid_c=args.seq_len, out_c=args.pre_len, n_heads=2, loc=slope_set.locs, args=args).to(args.device)

    # loss = torch.nn.MSELoss(reduction="mean")
    loss = torch.nn.MSELoss(reduction="mean")
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    num_nodes = slope_set.num_nodes
    args.num_nodes = num_nodes
    logger = utils.get_logger(logpath=os.path.join(args.save_dir, 'logs'), filepath=os.path.abspath(__file__))
    logger.info(args)

    if args.valid:
        validation_set = Subset(slope_set, range(int(slope_set.len * args.train), int(slope_set.len * (1 - args.test))))
        test_set = Subset(slope_set, range(int(slope_set.len * (1 - args.test)), slope_set.len))
        valid_loader = DataLoader(validation_set, batch_size=args.test_batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=True)

        trainer = GATTrainer(model, loss, optimizer, train_loader, valid_loader, test_loader, scaler, args)
    else:
        test_set = Subset(slope_set, range(int(slope_set.len * args.train), slope_set.len))
        test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=True)

        trainer = GATTrainer(model, loss, optimizer, train_loader, None, test_loader, scaler, args)

    if args.mode == "train":
        trainer.train()
    elif args.mode == "test":
        trainer.test(model, trainer.args, test_loader, scaler, path=args.save_dir + r"\best_model.pth")
    else:
        raise ValueError
