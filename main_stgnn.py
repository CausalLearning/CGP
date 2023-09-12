from __future__ import print_function
import sys
import os
import os.path as osp
import shutil
import time
import argparse
import logging
import hashlib
import copy
import csv
import numpy as np
import random

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch_geometric.transforms as T

from torch_geometric.datasets import Planetoid, Reddit, WebKB, Actor, WikipediaNetwork, Coauthor, Amazon, Flickr, WikiCS, Yelp
from torch_geometric.loader import NeighborLoader, RandomNodeSampler
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.nn.conv.gcn_conv import gcn_norm


from ogb.nodeproppred import Evaluator
from ogb.nodeproppred import PygNodePropPredDataset
from torch_scatter import scatter
from torch_geometric.utils import to_undirected, add_self_loops

import sparselearning
from models import initializers
from sparselearning.core import Masking, CosineDecay, LinearDecay
from models.model import GCNNet, SGCNet, APPNPNet, GCNIINet, GATNet, MLP, FAGCN, HGCN, LINK, GPRGNN, MixHop, FAGCNNet, HGCNNet

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

cudnn.benchmark = True
cudnn.deterministic = True


if not os.path.exists('./models'): os.mkdir('./models')
if not os.path.exists('./logs'): os.mkdir('./logs')
if not os.path.exists('./results'): os.mkdir('./results')
logger = None

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

models = {}
models['gcn'] = (GCNNet)
models['gat'] = (GATNet)
models['sgc'] = (SGCNet)
models['appnp'] = (APPNPNet)
models['gcnii'] = (GCNIINet)
models['mlp'] = (MLP)
models['fagcn'] = (FAGCN)
models['h2gcn'] = (HGCN)
models['link'] = (LINK)
models['gprgnn'] = (GPRGNN)
models['mixhop'] = (MixHop)
models['fagcnnet'] = (FAGCNNet)
models['h2gcnnet'] = (HGCNNet)


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    print("SAVING")
    torch.save(state, filename)


def setup_logger(args):
    global logger
    if logger == None:
        logger = logging.getLogger()
    else:  # wish there was a logger.close()
        for handler in logger.handlers[:]:  # make a copy of the list
            logger.removeHandler(handler)

    args_copy = copy.deepcopy(args)
    # copy to get a clean hash
    # use the same log file hash if iterations or verbose are different
    # these flags do not change the results
    args_copy.iters = 1
    args_copy.verbose = False
    args_copy.log_interval = 1
    args_copy.seed = 0

    if args.weight_sparse and not args.adj_sparse and not args.feature_sparse  :
        sparse_way = 'w'
        log_path = './logs/{0}/{1}_{2}_{3}_{4}.log'.format(
            sparse_way,args.model,args.data, args.final_density, hashlib.md5(str(args_copy).encode('utf-8')).hexdigest()[:8])
    elif args.adj_sparse and not args.weight_sparse and not args.feature_sparse :
        sparse_way = 'a'
        log_path = './logs/{0}/{1}_{2}_{3}_{4}.log'.format(
            sparse_way,args.model,args.data, args.final_density_adj, hashlib.md5(str(args_copy).encode('utf-8')).hexdigest()[:8])
    elif args.feature_sparse and not args.weight_sparse and not args.adj_sparse :
        sparse_way = 'f'
        log_path = './logs/{0}/{1}_{2}_{3}_{4}.log'.format(
            sparse_way,args.model,args.data, args.final_density_feature, hashlib.md5(str(args_copy).encode('utf-8')).hexdigest()[:8])
    elif args.weight_sparse and args.adj_sparse and not args.feature_sparse :
        sparse_way = 'wa'
        log_path = './logs/{0}/{1}_{2}_{3}_{4}_{5}.log'.format(
            sparse_way,args.model,args.data, args.final_density, args.final_density_adj, hashlib.md5(str(args_copy).encode('utf-8')).hexdigest()[:8])
    elif args.weight_sparse and args.feature_sparse and not args.adj_sparse :
        sparse_way = 'wf'
        log_path = './logs/{0}/{1}_{2}_{3}_{4}_{5}.log'.format(
            sparse_way,args.model,args.data, args.final_density, args.final_density_feature, hashlib.md5(str(args_copy).encode('utf-8')).hexdigest()[:8])
    elif args.adj_sparse and args.feature_sparse and not args.weight_sparse :
        sparse_way = 'af'
        log_path = './logs/{0}/{1}_{2}_{3}_{4}_{5}.log'.format(
            sparse_way,args.model,args.data, args.final_density_adj, args.final_density_feature, hashlib.md5(str(args_copy).encode('utf-8')).hexdigest()[:8])
    elif args.weight_sparse and  args.adj_sparse and args.feature_sparse:
        sparse_way = 'waf'
        log_path = './logs/{0}/{1}_{2}_{3}_{4}_{5}_{6}.log'.format(
            sparse_way,args.model,args.data, args.final_density, args.final_density_adj, args.final_density_feature, hashlib.md5(str(args_copy).encode('utf-8')).hexdigest()[:8])
    else:
        sparse_way = 'base'
        log_path = './logs/{0}/{1}_{2}_{3}.log'.format(
            sparse_way,args.model,args.data, hashlib.md5(str(args_copy).encode('utf-8')).hexdigest()[:8])

    if not os.path.exists('./logs/{}'.format(sparse_way)): os.mkdir('./logs/{}'.format(sparse_way))

    #log_path = './logs/{0}/{1}_{2}_{3}_{4}_{5}.log'.format(sparse_way, args.model, args.data, args.final_density, args.final_density_adj, hashlib.md5(str(args_copy).encode('utf-8')).hexdigest()[:8])

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s: %(message)s', datefmt='%H:%M:%S')

    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

def print_and_log(msg):
    global logger
    print(msg)
    logger.info(msg)


def results_to_file(args, train_acc, val_acc, test_acc, train_time, test_time):


    if args.weight_sparse and not args.adj_sparse and not args.feature_sparse  :
        sparse_way = 'w'
        filename = "./results/{}/{}_{}_{}_{}_result.csv".format(
                            sparse_way, args.model, args.data, args.init_density, args.final_density)
    elif args.adj_sparse and not args.weight_sparse and not args.feature_sparse :
        sparse_way = 'a'
        filename = "./results/{}/{}_{}_{}_{}_result.csv".format(
                            sparse_way, args.model, args.data, args.final_density_adj)
    elif args.feature_sparse and not args.weight_sparse and not args.adj_sparse :
        sparse_way = 'f'
        filename = "./results/{}/{}_{}_{}_{}_result.csv".format(
                            sparse_way, args.model, args.data, args.final_density_feature)
    elif args.weight_sparse and args.adj_sparse and not args.feature_sparse :
        sparse_way = 'wa'
        filename = "./results/{}/{}_{}_{}_{}_result.csv".format(
                            sparse_way, args.model, args.data, args.final_density, args.final_density_adj)
    elif args.weight_sparse and args.feature_sparse and not args.adj_sparse :
        sparse_way = 'wf'
        filename = "./results/{}/{}_{}_{}_{}_result.csv".format(
                            sparse_way, args.model, args.data, args.final_density, args.final_density_feature)
    elif args.adj_sparse and args.feature_sparse and not args.weight_sparse :
        sparse_way = 'af'
        filename = "./results/{}/{}_{}_{}_{}_result.csv".format(
                            sparse_way, args.model, args.data, args.final_density_adj, args.final_density_feature)
    elif args.weight_sparse and  args.adj_sparse and args.feature_sparse:
        sparse_way = 'waf'
        filename = "./results/{}/{}_{}_{}_{}_{}_result.csv".format(
                            sparse_way, args.model, args.data, args.final_density, args.final_density_adj, args.final_density_feature)
    else:
        sparse_way = 'base'
        filename = "./results/{}/{}_{}_result.csv".format(
                            sparse_way, args.model, args.data)

    if not os.path.exists('./results/{}'.format(sparse_way)): os.mkdir('./results/{}'.format(sparse_way))

    headerList = ["Method","Growth","Prune Rate", "Update Frequency", "Final Prune Epoch", "::", "train_acc", "val_acc", "test_acc", "train_time", "test_time"]

    #filename = "./results/{}/{}_{}_{}_{}_result.csv".format(sparse_way, args.model, args.data, args.final_density, args.final_density_adj)
    with open(filename, "a+") as f:

        # reader = csv.reader(f)
        # row1 = next(reader)
        f.seek(0)
        header = f.read(6)
        if  header != "Method":
            dw = csv.DictWriter(f, delimiter=',',
                        fieldnames=headerList)
            dw.writeheader()

        line = "{}, {}, {}, {}, {}, :::,   {:.4f}, {:.4f}, {:.4f},{:.4f}, {:.4f}\n".format(
            args.method, args.growth_schedule, args.prune_rate, args.update_frequency, args.final_prune_epoch, train_acc, val_acc, test_acc, train_time, test_time
        )
        f.write(line)



def train(args, model, device, data, optimizer, epoch, mask=None):
    model.train()
    train_loss = 0
    correct = 0
    n = 0
    criterion = torch.nn.BCEWithLogitsLoss()

    data = data.to(device)

    target = data.y[data.train_mask].to(device)


    if args.fp16: data = data.half()

    optimizer.zero_grad()

    output = model(data)[data.train_mask]
    if args.data in ['ogbn-proteins']:
        loss = criterion(output, target)
        acc  = 0.0
    else:
        loss = F.nll_loss(output, target)
        pred = output.max(1)[1]
        acc = pred.eq(data.y[data.train_mask]).sum().item() / data.train_mask.sum().item()

    if args.fp16:
        optimizer.backward(loss)
    else:
        loss.backward()

    if mask is not None:
        #print("Mask!!!!")
        mask.step()
    else:
        optimizer.step()

    # print_and_log('\n{}: Average loss: {:.4f}, Accuracy: {} \n'.format(
    #     'Training summary', loss, acc,))

def evaluate(args, model, device, data, is_test_set=False):
    model.eval()
    test_loss = 0
    correct = 0
    n = 0
    with torch.no_grad():
        #target = data.y[data.train_mask].to(device)
        data = data.to(device)

        if args.fp16: data = data.half()

        logits, accs = model(data), []

        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)
        train_acc, val_acc, tmp_test_acc = accs

    # print_and_log('\n{}: Train Accuracy: {:.4f}, Val Accuracy: {} Test Accuracy: {}\n'.format(
    #     'Test evaluation' if is_test_set else 'Evaluation',
    #     train_acc, val_acc, tmp_test_acc))
    return train_acc, val_acc, tmp_test_acc

def evaluate_ogb(args, model, device, data, evaluator, is_test_set=False):

    model.eval()
    test_loss = 0
    correct = 0
    n = 0
    with torch.no_grad():
        #target = data.y[data.train_mask].to(device)
        data = data.to(device)

        if args.fp16: data = data.half()

        out = model(data)

        if args.data in ['ogbn-arxiv','ogbn-products' ]:
            y_pred = out.argmax(dim=-1, keepdim=True)

            train_acc = evaluator.eval({
                            'y_true': data.y.unsqueeze(-1)[data.train_mask],
                            'y_pred': y_pred[data.train_mask],
                                    })['acc']
            val_acc = evaluator.eval({
                            'y_true': data.y.unsqueeze(-1)[data.valid_mask],
                            'y_pred': y_pred[data.valid_mask],
                                    })['acc']
            tmp_test_acc = evaluator.eval({
                            'y_true': data.y.unsqueeze(-1)[data.test_mask],
                            'y_pred': y_pred[data.test_mask],
                                    })['acc']
            print_and_log('\n{}: Train Accuracy: {:.4f}, Val Accuracy: {} Test Accuracy: {}\n'.format(
                            'Test evaluation' if is_test_set else 'Evaluation',
                            train_acc, val_acc, tmp_test_acc))

        elif args.data in ['ogbn-proteins']:

            train_acc = evaluator.eval({
                            'y_true': data.y[data.train_mask],
                            'y_pred': out[data.train_mask],
                                    })['rocauc'] # Acutually roc-auc, only name it train_acc
            val_acc = evaluator.eval({
                            'y_true': data.y[data.valid_mask],
                            'y_pred': out[data.valid_mask],
                                    })['rocauc']
            tmp_test_acc = evaluator.eval({
                            'y_true': data.y[data.test_mask],
                            'y_pred': out[data.test_mask],
                                    })['rocauc']
            print_and_log('\n{}: Train ROC-AUC: {:.4f}, Val ROC-AUC: {} Test ROC-AUC: {}\n'.format(
                            'Test evaluation' if is_test_set else 'Evaluation',
                             train_acc, val_acc, tmp_test_acc))

    return train_acc, val_acc, tmp_test_acc


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch GraNet for sparse training')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--batch-size-jac', type=int, default=200, metavar='N',
                        help='batch size for jac (default: 1000)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--multiplier', type=int, default=1, metavar='N',
                        help='extend training time by multiplier times')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=17, metavar='S', help='random seed (default: 17)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--optimizer', type=str, default='adam', help='The optimizer to use. Default: sgd. Options: sgd, adam.')
    randomhash = ''.join(str(time.time()).split('.'))
    parser.add_argument('--save', type=str, default=randomhash + '.pt',
                        help='path to save the final model')
    parser.add_argument('--data', type=str, default='mnist')
    parser.add_argument('--decay_frequency', type=int, default=25000)
    parser.add_argument('--l1', type=float, default=0.0)
    parser.add_argument('--fp16', action='store_true', help='Run in fp16 mode.')
    parser.add_argument('--valid_split', type=float, default=0.1)
    parser.add_argument('--resume', type=str)
    parser.add_argument('--start-epoch', type=int, default=1)
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--l2', type=float, default=1.0e-4)
    parser.add_argument('--iters', type=int, default=1, help='How many times the model should be run after each other. Default=1')
    parser.add_argument('--save-features', action='store_true', help='Resumes a saved model and saves its feature data to disk for plotting.')
    parser.add_argument('--bench', action='store_true', help='Enables the benchmarking of layers and estimates sparse speedups')
    parser.add_argument('--max-threads', type=int, default=10, help='How many threads to use for data loading.')
    parser.add_argument('--decay-schedule', type=str, default='cosine', help='The decay schedule for the pruning rate. Default: cosine. Choose from: cosine, linear.')
    parser.add_argument('--growth_schedule', type=str, default='gradient', help='The growth schedule. Default: gradient. Choose from: gradient, momentum, random.')
    parser.add_argument('--lr_scheduler', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--adj_sparse', action='store_true', help='If Sparse Adj.')
    parser.add_argument('--feature_sparse', action='store_true', help='If Sparse Weight.')
    parser.add_argument('--weight_sparse', action='store_true', help='If Sparse Feature.',)
    parser.add_argument('--dim', type=int, default=512)
    parser.add_argument('--cuda', type=int, default=0)


    # FAGCN
    parser.add_argument('--fagcn_layer_num', type=int, default=1)
    parser.add_argument('--fagcn_dropout', type=float, default=0)
    parser.add_argument('--fagcn_eps', type=float, default=0.1)

    # MixHop
    parser.add_argument('--mixhop_layer_num', type=int, default=1)
    parser.add_argument('--mixhop_dropout', type=float, default=0)
    parser.add_argument('--mixhop_hop', type=int, default=2)

    # GPRGNN
    parser.add_argument('--gprgnn_alpha', type=float, default=0.1)
    parser.add_argument('--gprgnn_k', type=int, default=10)

    # H2GCN

    parser.add_argument('--h2gcn_dropout', type=float, default=0.1)


    sparselearning.core.add_sparse_args(parser)

    args = parser.parse_args()
    setup_logger(args)
    print_and_log(args)

    if args.fp16:
        try:
            from apex.fp16_utils import FP16_Optimizer
        except:
            print('WARNING: apex not installed, ignoring --fp16 option')
            args.fp16 = False

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device('cuda:{}'.format(args.cuda) if use_cuda else "cpu")



    print_and_log('\n\n')
    print_and_log('='*80)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    for i in range(args.iters):

        #######################################################################################
        ############################# Datasets ################################################
        #######################################################################################
        print_and_log("\nIteration start: {0}/{1}\n".format(i+1, args.iters))


        if args.data in ['cora','citeseer','pubmed']:
            path = osp.join(osp.dirname(osp.realpath(__file__)), '../data', args.data)
            dataset = Planetoid(path, args.data, transform=T.NormalizeFeatures())
            data = dataset[0]

            data.num_classes = dataset.num_classes
            data.num_edges_orig = data.num_edges
            #print_and_log(data)
            #raise Exception('pause!! ')

        elif args.data in ["Cornell", "Texas", "Wisconsin"] :
            path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data', args.data)
            dataset = WebKB(path,args.data, transform=T.NormalizeFeatures())
            data = dataset[0]
            data.num_classes = dataset.num_classes
            print_and_log(data)
            data.train_mask = data.train_mask[:, args.seed % 10]
            data.val_mask = data.val_mask[:, args.seed % 10]
            data.test_mask = data.test_mask[:, args.seed % 10]
            #print_and_log(data)
            #raise Exception('pause!! ')

        elif args.data in ["Actor"] :
            path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data', args.data)
            dataset = Actor(path, transform=T.NormalizeFeatures())
            data = dataset[0]
            data.num_classes = dataset.num_classes

            #print_and_log(data)
            data.train_mask = data.train_mask[:, args.seed % 10]
            data.val_mask = data.val_mask[:, args.seed % 10]
            data.test_mask = data.test_mask[:, args.seed % 10]
            #print_and_log(data)
            #raise Exception('pause!! ')

        elif args.data in ["chameleon", "crocodile", "squirrel"] :
            path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data', args.data)
            dataset = WikipediaNetwork(path, args.data, transform=T.NormalizeFeatures())
            data = dataset[0]
            data.num_classes = dataset.num_classes

            data.train_mask = data.train_mask[:, args.seed % 10]
            data.val_mask = data.val_mask[:, args.seed % 10]
            data.test_mask = data.test_mask[:, args.seed % 10]
            #print_and_log(data)

            # Multi Spilt
            raise Exception('pause!! ')


        elif args.data in ["CS", "Physics"] :
            path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data', args.data)
            dataset = Coauthor(path, args.data, transform=T.NormalizeFeatures())
            data = dataset[0]
            data.num_classes = dataset.num_classes
            transform = RandomNodeSplit(split= "test_rest",
                                        num_train_per_class = 20,
                                        num_val = 30* data.num_classes,)
            transform(data)
            #print_and_log(data)
            #raise Exception('pause!! ')

        elif args.data in ["Computers", "Photo"] :
            path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data', args.data)
            dataset = Amazon(path, args.data, transform=T.NormalizeFeatures())
            data = dataset[0]
            data.num_classes = dataset.num_classes
            transform = RandomNodeSplit(split= "test_rest",
                                        num_train_per_class = 20,
                                        num_val = 30* data.num_classes,)
            transform(data)
            #print_and_log(data)

            #raise Exception('pause!! ')


        elif args.data in ["Flickr"] :
            path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data', args.data)
            dataset = Flickr(path)
            data = dataset[0]
            data.num_classes = dataset.num_classes
            print_and_log(data)
            # Cannot load file containing pickled data when allow_pickle=False
            raise Exception('pause!! ')

        elif args.data in ["Yelp"] :
            path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data', args.data)
            dataset = Yelp(path)
            data = dataset[0]
            data.num_classes = dataset.num_classes
            print_and_log(data)
            # Cannot load file containing pickled data when allow_pickle=False
            # Fix: allow_pickle=True)
            raise Exception('pause!! ')


        elif args.data in ["WikiCS"] :
            path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data', args.data)
            dataset = WikiCS(path, transform=T.NormalizeFeatures())
            data = dataset[0]
            data.num_classes = dataset.num_classes

            data.stopping_mask = None
            data.train_mask = data.train_mask[:, args.seed % 20]
            data.val_mask = data.val_mask[:, args.seed % 20]
            print_and_log(data)

        elif args.data == 'reddit':
            print("Loading Reddit .....")
            path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data', 'Reddit')
            dataset = Reddit(path, transform=T.NormalizeFeatures())

            data = dataset[0]
            data.num_classes = dataset.num_classes
            #kwargs = {'batch_size': 1024, 'num_workers': 6, 'persistent_workers': True}
            print_and_log(data)

            print("Load Reddit Done!")

        elif args.data in['ogbn-arxiv', 'ogbn-products', 'ogbn-proteins', 'ogbn-papers100M']:

            print("Loading  Dataset: {}".format(args.data))

            dataset = PygNodePropPredDataset(name=args.data, root='../data')
            data = dataset[0]
            split_idx = dataset.get_idx_split()
            evaluator = Evaluator(args.data)

            edge_index = to_undirected(data.edge_index, data.num_nodes)
            #edge_index = add_self_loops(edge_index, num_nodes=data.num_nodes)[0]

            data.edge_index = edge_index
            for split in ['train', 'valid', 'test']:
                    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
                    mask[split_idx[split]] = True
                    data[f'{split}_mask'] = mask

            if args.data in ['ogbn-proteins']:
                data.y = data.y.to(torch.float)
                data.num_classes = dataset.num_tasks
                data.node_species = None
                row, col = data.edge_index
                data.x = scatter(data.edge_attr, col, 0, dim_size=data.num_nodes, reduce='add')
            else:
                data.num_classes = dataset.num_classes

            if args.data in ['ogbn-arxiv']:
                data.y = data.y.squeeze(1)

            print("Load Done !")
            print_and_log(data)

        #######################################################################################
        ############################# Models ################################################
        #######################################################################################

        if args.model not in models:
            print('You need to select an existing model via the --model argument. Available models include: ')
            for key in models:
                print('\t{0}'.format(key))
            raise Exception('You need to select a model')
        else:

            if args.model == 'gcn':
                model = GCNNet(data, args).to(args.device)

            elif args.model == 'sgc':
                model = SGCNet(data, args).to(args.device)

            elif args.model == 'appnp':
                model = APPNPNet(data, args).to(args.device)

            elif args.model == 'gat':
                model = GATNet(data, args).to(args.device)

            elif args.model == 'gcnii':
                model = GCNIINet(data, args).to(args.device)

            elif args.model == 'mlp':
                model = MLP(data, args).to(args.device)

            elif args.model == 'fagcn':
                model = FAGCN(data, args).to(args.device)

            elif args.model == 'h2gcn':
                model = HGCN(data, args).to(args.device)

            elif args.model == 'link':
                model = LINK(data, args).to(args.device)

            elif args.model == 'gprgnn':
                model = GPRGNN(data, args).to(args.device)

            elif args.model == 'mixhop':
                model = MixHop(data, args).to(args.device)

            elif args.model == 'fagcnnet':
                model = FAGCNNet(data, args).to(args.device)

            elif args.model == 'h2gcnnet':
                model = HGCNNet(data, args).to(args.device)

            else:
                cls, cls_args = models[args.model]
                if args.data == 'cifar100':
                    cls_args[2] = 100
                model = cls(*(cls_args + [args.save_features, args.bench])).to(args.device)
            print_and_log(model)
            print_and_log('='*60)
            print_and_log(args.model)
            print_and_log('='*60)

            print_and_log('='*60)
            print_and_log('Prune mode: {0}'.format(args.prune))
            print_and_log('Growth mode: {0}'.format(args.growth))
            print_and_log('Redistribution mode: {0}'.format(args.redistribution))
            print_and_log('='*60)


        optimizer = None
        if args.optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(),lr=args.lr,momentum=args.momentum,weight_decay=args.l2, nesterov=True)
        elif args.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.l2)
        else:
            print('Unknown optimizer: {0}'.format(args.optimizer))
            raise Exception('Unknown optimizer.')

        if args.lr_scheduler:
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.epochs / 2) * args.multiplier, int(args.epochs * 3 / 4) * args.multiplier], last_epoch=-1)
        else:
            lr_scheduler = None


        if args.resume:
            if os.path.isfile(args.resume):
                print_and_log("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                model.load_state_dict(checkpoint)
                original_acc = evaluate(args, model, args.device, test_loader)


        if args.fp16:
            print('FP16')
            optimizer = FP16_Optimizer(optimizer,
                                       static_loss_scale = None,
                                       dynamic_loss_scale = True,
                                       dynamic_loss_args = {'init_scale': 2 ** 16})
            model = model.half()


        mask = None
        if args.sparse:
            decay = CosineDecay(args.prune_rate, (args.epochs*args.multiplier))
            mask = Masking(optimizer, prune_rate=args.prune_rate, death_mode=args.prune, prune_rate_decay=decay, growth_mode=args.growth,
                           redistribution_mode=args.redistribution, args=args, train_loader=None, device =args.device)
            mask.add_module(model, sparse_init=args.sparse_init)

        best_acc = 0.0
        t_start = time.time()
        for epoch in range(1, args.epochs*args.multiplier + 1):

            #print_and_log("Epoch:{}".format(epoch))
            #print("="*50)

            # save models
            save_path = './save/' + str(args.model) + '/' + str(args.data) + '/' + str(args.method)  + '/' + str(args.seed)
            save_subfolder = os.path.join(save_path, 'Multiplier=' + str(args.multiplier)  + '_sparsity' + str(1-args.final_density))
            if not os.path.exists(save_subfolder): os.makedirs(save_subfolder)

            t0 = time.time()
            #print(mask)

            train(args, model, args.device, data, optimizer, epoch, mask)


            if lr_scheduler is not None:
                lr_scheduler.step()
            if args.valid_split > 0.0:
                if args.data in['ogbn-arxiv', 'ogbn-products', 'ogbn-proteins', 'ogbn-ogbn-papers100M']:
                    _, val_acc, _ = evaluate_ogb(args, model, args.device, data, evaluator)
                else:
                    _, val_acc, _ = evaluate(args, model, args.device, data)

            # target sparsity is reached
            if args.sparse:
                if epoch == args.multiplier * args.final_prune_epoch+1:
                    best_acc = 0.0

            if val_acc > best_acc:
                print('Saving model')
                best_acc = val_acc
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, filename=os.path.join(save_subfolder, 'model_final.pth'))

            #print_and_log(' Time taken for epoch: {:.2f} seconds.\n'.format(time.time() - t0))

        train_time_total = time.time() - t_start
        print('Testing model')
        model.load_state_dict(torch.load(os.path.join(save_subfolder, 'model_final.pth'))['state_dict'])

        t_test_0 = time.time()
        if args.data in['ogbn-arxiv', 'ogbn-products', 'ogbn-proteins', 'ogbn-ogbn-papers100M']:
            train_acc, val_acc, test_acc = evaluate_ogb(args, model, args.device, data, evaluator,  is_test_set=True)
        else:
            train_acc, val_acc, test_acc = evaluate(args, model, args.device, data, is_test_set=True)
        print('Test accuracy is:', test_acc)
        results_to_file(args, train_acc, val_acc, test_acc, train_time_total, time.time()- t_test_0)



if __name__ == '__main__':
    print("Start Runing!")
    main()
