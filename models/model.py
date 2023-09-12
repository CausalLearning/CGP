import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import ChebConv, GCNConv, SGConv, APPNP, GCN2Conv, JumpingKnowledge, MessagePassing # noqa
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from models.gat_conv import GATConv
from models.fagcn_conv import FAConv

from torch_geometric.utils import to_scipy_sparse_matrix
import torch_sparse
from torch_sparse import SparseTensor, matmul
import scipy.sparse
import numpy as np



class GCNNet(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GCNNet, self).__init__()
        
        self.args = args
        self.conv1 = GCNConv(dataset.num_features, args.dim, cached=False, add_self_loops = True, normalize = True)
        self.conv2 = GCNConv(args.dim, dataset.num_classes, cached=False, add_self_loops = True, normalize = True)

        if args.adj_sparse:
            self.edge_weight_train = nn.Parameter(torch.randn_like(dataset.edge_index[0].to(torch.float32)))

        if args.feature_sparse:
            self.x_weight = nn.Parameter(torch.ones_like(dataset.x)[0].to(torch.float32))

    def forward(self, data, data_mask=None):
        
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        if self.args.adj_sparse:
            edge_mask = torch.abs(self.edge_weight_train) > 0
            row, col = data.edge_index
            row, col= row[edge_mask], col[edge_mask]
            edge_index = torch.stack([row, col], dim=0)
            edge_weight = self.edge_weight_train.sigmoid()[edge_mask]

        if self.args.feature_sparse:
            x_mask = (torch.abs(self.x_weight) > 0).float()
            x_weight = torch.mul(self.x_weight.sigmoid(), x_mask)
            x = torch.mul(x,x_weight)


        x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        return F.log_softmax(x, dim=1)


class GATNet(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GATNet, self).__init__()

        self.args = args
        self.conv1 = GATConv(dataset.num_features, 8, heads=8)
        self.conv2 = GATConv(8 * 8, dataset.num_classes, heads=1, concat=False)

        if args.adj_sparse:
            self.edge_weight_train = nn.Parameter(torch.randn_like(dataset.edge_index[0].to(torch.float32)))

        if args.feature_sparse:
            self.x_weight = nn.Parameter(torch.ones_like(dataset.x)[0].to(torch.float32))


    def forward(self, data, data_mask=None):

        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        if self.args.adj_sparse:
            edge_mask = torch.abs(self.edge_weight_train) > 0
            row, col = data.edge_index
            row, col= row[edge_mask], col[edge_mask]
            edge_index = torch.stack([row, col], dim=0)
            edge_weight = self.edge_weight_train.sigmoid()[edge_mask]

        if self.args.feature_sparse:
            x_mask = (torch.abs(self.x_weight) > 0).float()
            x_weight = torch.mul(self.x_weight.sigmoid(), x_mask)
            x = torch.mul(x,x_weight)

        x = F.dropout(x, p=0.5, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        return x.log_softmax(dim=-1)


class SGCNet(torch.nn.Module):
    def __init__(self, dataset, args):
        super().__init__()

        self.args = args
        self.conv1 = SGConv(dataset.num_features, dataset.num_classes, K=2,
                            cached=True)
        if args.adj_sparse:
            self.edge_weight_train = nn.Parameter(torch.randn_like(dataset.edge_index[0].to(torch.float32)))

        if args.feature_sparse:
            self.x_weight = nn.Parameter(torch.ones_like(dataset.x)[0].to(torch.float32))

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        if self.args.adj_sparse:
            edge_mask = torch.abs(self.edge_weight_train) > 0
            row, col = data.edge_index
            row, col= row[edge_mask], col[edge_mask]
            edge_index = torch.stack([row, col], dim=0)
            edge_weight = self.edge_weight_train.sigmoid()[edge_mask]

        if self.args.feature_sparse:
            x_mask = (torch.abs(self.x_weight) > 0).float()
            x_weight = torch.mul(self.x_weight.sigmoid(), x_mask)
            x = torch.mul(x,x_weight)

        x = self.conv1(x, edge_index, edge_weight= edge_weight)
        return F.log_softmax(x, dim=1)


class APPNPNet(torch.nn.Module):
    def __init__(self, dataset, args):
        super().__init__()
        self.args = args
        self.lin1 = Linear(dataset.num_features, args.dim)
        self.lin2 = Linear(args.dim, dataset.num_classes)
        self.prop1 = APPNP(10, 0.1)

        if args.adj_sparse:
            self.edge_weight_train = nn.Parameter(torch.randn_like(dataset.edge_index[0].to(torch.float32)))

        if args.feature_sparse:
            self.x_weight = nn.Parameter(torch.ones_like(dataset.x)[0].to(torch.float32))



    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        if self.args.adj_sparse:
            edge_mask = torch.abs(self.edge_weight_train) > 0
            row, col = data.edge_index
            row, col= row[edge_mask], col[edge_mask]
            edge_index = torch.stack([row, col], dim=0)
            edge_weight = self.edge_weight_train.sigmoid()[edge_mask]

        if self.args.feature_sparse:
            x_mask = (torch.abs(self.x_weight) > 0).float()
            x_weight = torch.mul(self.x_weight.sigmoid(), x_mask)
            x = torch.mul(x,x_weight)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index, edge_weight=edge_weight)
        return F.log_softmax(x, dim=1)

class GCNIINet(torch.nn.Module):
    def __init__(self, dataset, args,):
        super().__init__()
        # alpha =0.1
        # theta =0.5
        self.args = args
        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(dataset.num_features, args.dim))
        self.lins.append(Linear(args.dim, dataset.num_classes))

        self.convs = torch.nn.ModuleList()
        for layer in range(20):
            self.convs.append(
                GCN2Conv(args.dim, 0.1, 0.5, layer + 1,
                         shared_weights =True, normalize=True))

        if args.adj_sparse:
            self.edge_weight_train = nn.Parameter(torch.randn_like(dataset.edge_index[0].to(torch.float32)))

        if args.feature_sparse:
            self.x_weight = nn.Parameter(torch.ones_like(dataset.x)[0].to(torch.float32))



    def forward(self, data):


        x, edge_index = data.x, data.edge_index

        if self.args.feature_sparse:
            x_mask = (torch.abs(self.x_weight) > 0).float()
            x_weight = torch.mul(self.x_weight.sigmoid(), x_mask)
            x = torch.mul(x,x_weight)

        x = F.dropout(x, training=self.training)
        x = x_0 = self.lins[0](x).relu()

        if self.args.adj_sparse:
            edge_mask = torch.abs(self.edge_weight_train) > 0
            row, col = data.edge_index
            row, col= row[edge_mask], col[edge_mask]
            edge_index = torch.stack([row, col], dim=0)
            edge_weight = self.edge_weight_train.sigmoid()[edge_mask]





        for conv in self.convs:
            x = F.dropout(x, training=self.training)
            x = conv(x, x_0, edge_index, edge_weight= edge_weight)
            x = x.relu()

        x = F.dropout(x, training=self.training)
        x = self.lins[1](x)

        return x.log_softmax(dim=-1)


class MLP(nn.Module):
    """

    """
    def __init__(self, dataset, args):
        super(MLP,self).__init__()

        self.num_layers=2
        self.dropout_rate=0.5

        self.lins=nn.ModuleList()
        self.lins.append(nn.Linear(dataset.num_features, args.dim))
        for i in range(self.num_layers-2):
            self.lins.append(nn.Linear(args.dim, args.dim))
        self.lins.append(nn.Linear(args.dim,dataset.num_classes))

        self.bns=nn.ModuleList()
        for i in range(self.num_layers-1):
            self.bns.append(nn.BatchNorm1d(args.dim))
        self.bns.append(nn.BatchNorm1d(dataset.num_classes))

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        for i in range(self.num_layers-1):
            x=self.lins[i](x)
            x=self.bns[i](x)
        x=self.lins[self.num_layers-1](x)

        return F.log_softmax(x, dim=1)


class LINK(nn.Module):
    """ logistic regression on adjacency matrix """

    def __init__(self, dataset, args):
        super(LINK, self).__init__()

        self.W = nn.Linear(dataset.x.size(0), dataset.num_classes)

    def reset_parameters(self):
        self.W.reset_parameters()

    def forward(self, data):
        N = data.x.size(0)
        edge_index = data.edge_index
        if isinstance(edge_index, torch.Tensor):
            row, col = edge_index
            A = SparseTensor(row=row, col=col, sparse_sizes=(N, N)).to_torch_sparse_coo_tensor()
        elif isinstance(edge_index, SparseTensor):
            A = edge_index.to_torch_sparse_coo_tensor()
        logits = self.W(A)
        return F.log_softmax(logits, dim=1)


class FAGCN(nn.Module):
    def __init__(self, dataset, args):
        super(FAGCN, self).__init__()
        self.eps = args.fagcn_eps
        self.layer_num = args.fagcn_layer_num
        self.dropout = args.fagcn_dropout

        self.layers = nn.ModuleList()
        for _ in range(self.layer_num):
            self.layers.append(FAConv(args.dim, self.eps, self.dropout))

        self.t1 = nn.Linear(dataset.num_features, args.dim)
        self.t2 = nn.Linear(args.dim, dataset.num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.t1.weight, gain=1.414)
        nn.init.xavier_normal_(self.t2.weight, gain=1.414)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        h = F.dropout(x, p=self.dropout, training=self.training)
        h = torch.relu(self.t1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        raw = h
        for i in range(self.layer_num):
            h = self.layers[i](h,raw,edge_index)
        h = self.t2(h)

        return F.log_softmax(h, dim=1)


class GPR_prop(MessagePassing):
    '''
    GPRGNN, from original repo https://github.com/jianhao2016/GPRGNN
    propagation class for GPR_GNN
    '''

    def __init__(self, K, alpha, Init, Gamma=None, bias=True, **kwargs):
        super(GPR_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            # SGC-like
            TEMP = 0.0*np.ones(K+1)
            TEMP[alpha] = 1.0
        elif Init == 'PPR':
            # PPR-like
            TEMP = alpha*(1-alpha)**np.arange(K+1)
            TEMP[-1] = (1-alpha)**K
        elif Init == 'NPPR':
            # Negative PPR
            TEMP = (alpha)**np.arange(K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'Random':
            # Random
            bound = np.sqrt(3/(K+1))
            TEMP = np.random.uniform(-bound, bound, K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'WS':
            # Specify Gamma
            TEMP = Gamma

        self.temp = nn.Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        nn.init.zeros_(self.temp)
        for k in range(self.K+1):
            self.temp.data[k] = self.alpha*(1-self.alpha)**k
        self.temp.data[-1] = (1-self.alpha)**self.K

    def forward(self, x, edge_index, edge_weight=None):
        if isinstance(edge_index, torch.Tensor):
            edge_index, norm = gcn_norm(
                edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(
                edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)
            norm = None

        hidden = x*(self.temp[0])
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            gamma = self.temp[k+1]
            hidden = hidden + gamma*x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)


class GPRGNN(nn.Module):
    """GPRGNN, from original repo https://github.com/jianhao2016/GPRGNN"""

    def __init__(self, dataset, args):
        super(GPRGNN, self).__init__()

        Init='PPR'
        dprate=.5
        dropout=.5
        K=args.gprgnn_k
        alpha= args.gprgnn_alpha
        Gamma=None
        ppnp='GPR_prop'
        self.lin1 = nn.Linear(dataset.num_features, args.dim)
        self.lin2 = nn.Linear(args.dim, dataset.num_classes)

        if ppnp == 'PPNP':
            self.prop1 = APPNP(K, alpha)
        elif ppnp == 'GPR_prop':
            self.prop1 = GPR_prop(K, alpha, Init, Gamma)

        self.Init = Init
        self.dprate = dprate
        self.dropout = dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prop1.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)

class MixHopLayer(nn.Module):
    """ Our MixHop layer """
    def __init__(self, in_channels, out_channels, hops=2):
        super(MixHopLayer, self).__init__()
        self.hops = hops
        self.lins = nn.ModuleList()
        for hop in range(self.hops+1):
            lin = nn.Linear(in_channels, out_channels)
            self.lins.append(lin)

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x, adj_t):
        xs = [self.lins[0](x) ]
        for j in range(1,self.hops+1):
            # less runtime efficient but usually more memory efficient to mult weight matrix first
            x_j = self.lins[j](x)
            for hop in range(j):
                x_j = matmul(adj_t, x_j)
            xs += [x_j]
        return torch.cat(xs, dim=1)

class MixHop(nn.Module):
    """ our implementation of MixHop
    some assumptions: the powers of the adjacency are [0, 1, ..., hops],
        with every power in between
    each concatenated layer has the same dimension --- hidden_channels
    """
    def __init__(self, dataset, args):
        super(MixHop, self).__init__()

        num_layers= args.mixhop_layer_num
        dropout=args.mixhop_dropout
        hops=args.mixhop_hop

        self.convs = nn.ModuleList()
        self.convs.append(MixHopLayer(dataset.num_features, args.dim, hops=hops))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(args.dim*(hops+1)))
        for _ in range(num_layers - 2):
            self.convs.append(
                MixHopLayer(args.dim*(hops+1), args.dim, hops=hops))
            self.bns.append(nn.BatchNorm1d(args.dim*(hops+1)))

        self.convs.append(
            MixHopLayer(args.dim*(hops+1), dataset.num_classes, hops=hops))

        # note: uses linear projection instead of paper's attention output
        self.final_project = nn.Linear(dataset.num_classes*(hops+1), dataset.num_classes)

        self.dropout = dropout
        self.activation = F.relu

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.final_project.reset_parameters()


    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        n = data.x.size(0)
        edge_weight = None

        if isinstance(edge_index, torch.Tensor):
            edge_index, edge_weight = gcn_norm(
                edge_index, edge_weight, n, False,
                 dtype=x.dtype)
            row, col = edge_index
            adj_t = SparseTensor(row=col, col=row, value=edge_weight, sparse_sizes=(n, n))
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(
                edge_index, edge_weight, n, False,
                dtype=x.dtype)
            edge_weight=None
            adj_t = edge_index

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)

        x = self.final_project(x)
        return x


class HGCN(nn.Module):
    def __init__(self,dataset, args):
        super(HGCN, self).__init__()
        self.args = args
        self.lin1 = nn.Linear(dataset.num_features, args.dim)
        self.lin = nn.Linear(args.dim*5,dataset.num_classes)

    def forward(self,data):

        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        edge_index, edge_weight = gcn_norm(edge_index, edge_weight)

        adj_coo = to_scipy_sparse_matrix(edge_index)
        adj_row = adj_coo.row
        adj_col = adj_coo.col
        adj_value = edge_weight.detach().cpu().numpy()
        # print(adj_value)
        # raise exception("pause")
        adj_size = adj_coo.shape
        edge_index = torch_sparse.SparseTensor(sparse_sizes=[adj_size[0], adj_size[1]], row=torch.tensor(adj_row, dtype=torch.long),
                                        col=torch.tensor(adj_col, dtype=torch.long),
                                        value=torch.tensor(adj_value, dtype=torch.float32)).to(self.args.device)
        temp = self.lin1(x)
        temp = F.relu(temp)
        temp1 = torch_sparse.matmul(edge_index,temp)
        temp1 =torch.cat((temp,temp1),dim=1)
        temp2 = torch_sparse.matmul(edge_index,temp1)
        temp = torch.cat((temp,temp1,temp2),dim=1)
        temp = F.dropout(temp,p=self.args.h2gcn_dropout)
        ans = self.lin(temp)

        return F.log_softmax(ans, dim=1)


class FAGCNNet(nn.Module):
    def __init__(self, dataset, args):
        super(FAGCNNet, self).__init__()
        self.eps = 0.3
        self.layer_num = 2
        self.dropout = 0.6
        self.args = args

        self.layers = nn.ModuleList()
        for _ in range(self.layer_num):
            self.layers.append(FAConv(args.dim, self.eps, self.dropout))

        self.t1 = nn.Linear(dataset.num_features, args.dim)
        self.t2 = nn.Linear(args.dim, dataset.num_classes)
        self.reset_parameters()

        if args.adj_sparse:
            self.edge_weight_train = nn.Parameter(torch.randn_like(dataset.edge_index[0].to(torch.float32)))

    def reset_parameters(self):
        nn.init.xavier_normal_(self.t1.weight, gain=1.414)
        nn.init.xavier_normal_(self.t2.weight, gain=1.414)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        if self.args.adj_sparse:
            edge_mask = torch.abs(self.edge_weight_train) > 0
            row, col = data.edge_index
            row, col= row[edge_mask], col[edge_mask]
            edge_index = torch.stack([row, col], dim=0)
            edge_weight = self.edge_weight_train.sigmoid()[edge_mask]


        h = F.dropout(x, p=self.dropout, training=self.training)
        h = torch.relu(self.t1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        raw = h
        for i in range(self.layer_num):
            h = self.layers[i](h,raw,edge_index, edge_weight)
        h = self.t2(h)

        return F.log_softmax(h, dim=1)



class HGCNNet(nn.Module):
    def __init__(self,dataset, args):
        super(HGCNNet, self).__init__()
        self.args = args
        self.lin1 = nn.Linear(dataset.num_features, args.dim)
        self.lin = nn.Linear(args.dim*5,dataset.num_classes)

        if args.adj_sparse:
            self.edge_weight_train = nn.Parameter(torch.randn_like(dataset.edge_index[0].to(torch.float32)))


    def forward(self,data):

        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        if self.args.adj_sparse:
            edge_mask = torch.abs(self.edge_weight_train) > 0
            row, col = data.edge_index
            row, col= row[edge_mask], col[edge_mask]
            edge_index = torch.stack([row, col], dim=0)
            edge_weight = self.edge_weight_train.sigmoid()[edge_mask]

        edge_index, edge_weight = gcn_norm(edge_index, edge_weight, num_nodes = x.size(0))

        adj_coo = to_scipy_sparse_matrix(edge_index)
        adj_row = adj_coo.row
        adj_col = adj_coo.col
        adj_value = edge_weight.detach().cpu().numpy()
        # print(adj_value)
        # raise exception("pause")
        adj_size = adj_coo.shape
        edge_index = torch_sparse.SparseTensor(sparse_sizes=[adj_size[0], adj_size[1]], row=torch.tensor(adj_row, dtype=torch.long),
                                        col=torch.tensor(adj_col, dtype=torch.long),
                                        value=torch.tensor(adj_value, dtype=torch.float32)).to(self.args.device)
        temp = self.lin1(x)
        temp = F.relu(temp)
        temp1 = torch_sparse.matmul(edge_index,temp)
        temp1 =torch.cat((temp,temp1),dim=1)
        temp2 = torch_sparse.matmul(edge_index,temp1)
        temp = torch.cat((temp,temp1,temp2),dim=1)
        temp = F.dropout(temp,p=0.5)
        ans = self.lin(temp)

        return F.log_softmax(ans, dim=1)




class GCNmasker(torch.nn.Module):
    
    def __init__(self, dataset, args):
        super(GCNmasker, self).__init__()

        self.conv1 = GCNConv(dataset.num_features, args.masker_dim, cached=False)
        self.conv2 = GCNConv(args.masker_dim, args.masker_dim, cached=False)
        self.mlp = nn.Linear(args.masker_dim * 2, 1)
        self.score_function = args.score_function
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, data):

        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        
        if self.score_function == 'inner_product':
            link_score = self.inner_product_score(x, edge_index)
        elif self.score_function == 'concat_mlp':
            link_score = self.concat_mlp_score(x, edge_index)
        else:
            assert False

        return link_score
    
    def inner_product_score(self, x, edge_index):
        
        row, col = edge_index
        link_score = torch.sum(x[row] * x[col], dim=1)
        #print("max:{:.2f} min:{:.2f} mean:{:.2f}".format(link_score.max(), link_score.min(), link_score.mean()))
        link_score = self.sigmoid(link_score).view(-1)
        return link_score

    def concat_mlp_score(self, x, edge_index):
        
        row, col = edge_index
        link_score = torch.cat((x[row], x[col]), dim=1)
        link_score = self.mlp(link_score)
        # weight = self.mlp.weight
        # print("max:{:.2f} min:{:.2f} mean:{:.2f}".format(link_score.max(), link_score.min(), link_score.mean()))
        link_score = self.sigmoid(link_score).view(-1)
        return link_score