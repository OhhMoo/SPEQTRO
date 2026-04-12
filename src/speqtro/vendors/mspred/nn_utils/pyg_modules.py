"""pyg_modules.py -- vendored from ms-pred, PyG version.

Graph neural network modules using PyTorch Geometric.
Uses torch_scatter for message passing operations.

"""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch_scatter


class GatedGraphConv(nn.Module):
    r"""Gated Graph Convolution layer from `Gated Graph Sequence
    Neural Networks <https://arxiv.org/pdf/1511.05493.pdf>`__

    Supports multiple edge types via per-type linear transformations.

    Parameters
    ----------
    in_feats : int
        Input feature size.
    out_feats : int
        Output feature size.
    n_steps : int
        Number of recurrent steps.
    n_etypes : int
        Number of edge types.
    bias : bool
        If True, adds a learnable bias to the output. Default: ``True``.
    """

    def __init__(self, in_feats, out_feats, n_steps, n_etypes, bias=True):
        super(GatedGraphConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._n_steps = n_steps
        self._n_etypes = n_etypes
        self.linears = nn.ModuleList(
            [nn.Linear(out_feats, out_feats) for _ in range(n_etypes)]
        )
        self.gru = nn.GRUCell(out_feats, out_feats, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        gain = init.calculate_gain("relu")
        self.gru.reset_parameters()
        for linear in self.linears:
            init.xavier_normal_(linear.weight, gain=gain)
            init.zeros_(linear.bias)

    def forward(self, x, edge_index, etypes=None):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input node features of shape :math:`(N, D_{in})`.
        edge_index : torch.Tensor
            Edge indices of shape :math:`(2, E)`.
        etypes : torch.LongTensor or None
            Edge type tensor of shape :math:`(E,)`.

        Returns
        -------
        torch.Tensor
            Output node features of shape :math:`(N, D_{out})`.
        """
        N = x.shape[0]
        src, dst = edge_index[0], edge_index[1]

        if self._n_etypes != 1 and etypes is not None:
            assert (
                etypes.min() >= 0 and etypes.max() < self._n_etypes
            ), "edge type indices out of range [0, {})".format(self._n_etypes)

        zero_pad = x.new_zeros((N, self._out_feats - x.shape[1]))
        feat = torch.cat([x, zero_pad], -1)

        for _ in range(self._n_steps):
            if self._n_etypes == 1 and etypes is None:
                msg = self.linears[0](feat[src])
                a = torch_scatter.scatter(msg, dst, dim=0, reduce='sum', dim_size=N)
            else:
                msg = feat.new_zeros(edge_index.shape[1], self._out_feats)
                for i in range(self._n_etypes):
                    mask = (etypes == i)
                    if mask.any():
                        msg[mask] = self.linears[i](feat[src[mask]])
                a = torch_scatter.scatter(msg, dst, dim=0, reduce='sum', dim_size=N)
            feat = self.gru(a, feat)
        return feat


class PNAConvTower(nn.Module):
    """A single PNA tower in PNA layers"""

    def __init__(
        self,
        in_size,
        out_size,
        aggregators,
        scalers,
        delta,
        dropout=0.0,
        edge_feat_size=0,
    ):
        super(PNAConvTower, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.aggregators = aggregators
        self.scalers = scalers
        self.delta = delta
        self.edge_feat_size = edge_feat_size

        self.M = nn.Linear(2 * in_size + edge_feat_size, in_size)
        self.U = nn.Linear((len(aggregators) * len(scalers) + 1) * in_size, out_size)
        self.dropout = nn.Dropout(dropout)
        self.batchnorm = nn.BatchNorm1d(out_size)

    def forward(self, node_feat, edge_index, edge_feat=None, batch=None):
        src, dst = edge_index[0], edge_index[1]
        N = node_feat.shape[0]

        if batch is not None:
            batch_num_nodes = torch.bincount(batch)
            snorm_n = torch.cat(
                [torch.ones(n, 1, device=node_feat.device) / n for n in batch_num_nodes],
                dim=0,
            ).sqrt()
        else:
            snorm_n = torch.ones(N, 1, device=node_feat.device) / (N ** 0.5)

        if self.edge_feat_size > 0 and edge_feat is not None:
            f = torch.cat([node_feat[src], node_feat[dst], edge_feat], dim=-1)
        else:
            f = torch.cat([node_feat[src], node_feat[dst]], dim=-1)
        msg = self.M(f)

        agg_results = []
        for agg_name in self.aggregators:
            if agg_name in ('mean', 'sum', 'max', 'min'):
                agg_results.append(
                    torch_scatter.scatter(msg, dst, dim=0, reduce=agg_name, dim_size=N)
                )
            elif agg_name == 'var':
                mean_val = torch_scatter.scatter(msg, dst, dim=0, reduce='mean', dim_size=N)
                sq_val = torch_scatter.scatter(msg * msg, dst, dim=0, reduce='mean', dim_size=N)
                agg_results.append(torch.relu(sq_val - mean_val * mean_val))
            elif agg_name == 'std':
                mean_val = torch_scatter.scatter(msg, dst, dim=0, reduce='mean', dim_size=N)
                sq_val = torch_scatter.scatter(msg * msg, dst, dim=0, reduce='mean', dim_size=N)
                agg_results.append(torch.sqrt(torch.relu(sq_val - mean_val * mean_val) + 1e-30))

        h = torch.cat(agg_results, dim=1)

        degree = torch_scatter.scatter(
            torch.ones(len(src), device=node_feat.device), dst, dim_size=N, reduce='sum'
        )
        scaled = []
        for scaler in self.scalers:
            if scaler == 'identity':
                scaled.append(h)
            elif scaler == 'amplification':
                scaled.append(h * (torch.log(degree + 1) / self.delta).unsqueeze(-1))
            elif scaler == 'attenuation':
                scaled.append(h * (self.delta / (torch.log(degree + 1) + 1e-30)).unsqueeze(-1))
        h = torch.cat(scaled, dim=1)

        h = self.U(torch.cat([node_feat, h], dim=-1))
        h = h * snorm_n
        return self.dropout(self.batchnorm(h))


class PNAConv(nn.Module):
    r"""Principal Neighbourhood Aggregation Layer."""

    def __init__(
        self,
        in_size,
        out_size,
        aggregators,
        scalers,
        delta,
        dropout=0.0,
        num_towers=1,
        edge_feat_size=0,
        residual=True,
    ):
        super(PNAConv, self).__init__()

        self.in_size = in_size
        self.out_size = out_size
        assert in_size % num_towers == 0, "in_size must be divisible by num_towers"
        assert out_size % num_towers == 0, "out_size must be divisible by num_towers"
        self.tower_in_size = in_size // num_towers
        self.tower_out_size = out_size // num_towers
        self.edge_feat_size = edge_feat_size
        self.residual = residual
        if self.in_size != self.out_size:
            self.residual = False

        self.towers = nn.ModuleList(
            [
                PNAConvTower(
                    self.tower_in_size,
                    self.tower_out_size,
                    aggregators,
                    scalers,
                    delta,
                    dropout=dropout,
                    edge_feat_size=edge_feat_size,
                )
                for _ in range(num_towers)
            ]
        )

        self.mixing_layer = nn.Sequential(nn.Linear(out_size, out_size), nn.LeakyReLU())

    def forward(self, node_feat, edge_index, edge_feat=None, batch=None):
        h_cat = torch.cat(
            [
                tower(
                    node_feat[
                        :, ti * self.tower_in_size : (ti + 1) * self.tower_in_size
                    ],
                    edge_index,
                    edge_feat,
                    batch,
                )
                for ti, tower in enumerate(self.towers)
            ],
            dim=1,
        )
        h_out = self.mixing_layer(h_cat)
        if self.residual:
            h_out = h_out + node_feat

        return h_out


class GINEConv(nn.Module):
    r"""Graph Isomorphism Network with Edge Features."""

    def __init__(self, apply_func=None, init_eps=0, learn_eps=False):
        super(GINEConv, self).__init__()
        self.apply_func = apply_func
        if learn_eps:
            self.eps = nn.Parameter(torch.FloatTensor([init_eps]))
        else:
            self.register_buffer("eps", torch.FloatTensor([init_eps]))

    def forward(self, node_feat, edge_index, edge_feat):
        src, dst = edge_index[0], edge_index[1]
        N = node_feat.shape[0]
        msg = F.relu(node_feat[src] + edge_feat)
        neigh = torch_scatter.scatter(msg, dst, dim=0, reduce='sum', dim_size=N)
        rst = (1 + self.eps) * node_feat + neigh
        if self.apply_func is not None:
            rst = self.apply_func(rst)
        return rst


class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, x, edge_index):
        src, dst = edge_index[0], edge_index[1]
        N = x.shape[0]
        agg = torch_scatter.scatter(x[src], dst, dim=0, reduce='sum', dim_size=N)
        return self.linear(agg)


class MultiEdgeGCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, etypes):
        super().__init__()
        self.out_feats = out_feats
        self.etype_layers = nn.ModuleDict({
            etype: nn.Linear(in_feats, out_feats)
            for etype in etypes
        })

    def forward(self, x, edge_index, edge_types):
        N = x.shape[0]
        src, dst = edge_index[0], edge_index[1]
        outputs = torch.zeros(N, self.out_feats, device=x.device)
        for i, (etype, layer) in enumerate(self.etype_layers.items()):
            mask = (edge_types == i)
            if mask.any():
                msg = layer(x[src[mask]])
                outputs += torch_scatter.scatter(msg, dst[mask], dim=0, reduce='sum', dim_size=N)
        return outputs


class MultiEdgeGCN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, etypes, conv_steps):
        super().__init__()
        self.layer1 = MultiEdgeGCNLayer(in_feats, hidden_feats, etypes)
        self.layer2 = MultiEdgeGCNLayer(hidden_feats, out_feats, etypes)
        self.conv_steps = conv_steps

    def forward(self, x, edge_index, edge_types):
        h = x
        for _ in range(self.conv_steps-1):
            h = self.layer1(h, edge_index, edge_types)
            h = F.relu(h)
        h = self.layer2(h, edge_index, edge_types)
        return h


class HyperGNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_conv, dropout=0):
        super(HyperGNN, self).__init__()
        self.layer = GCNLayer(hidden_size, hidden_size)
        self.layer_out = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()
        self.dropout_conv = nn.Dropout(dropout)
        self.dropout_output = nn.Dropout(dropout)
        self.num_conv = num_conv

    def forward(self, x, edge_index):
        for _ in range(self.num_conv):
            x = self.layer(x, edge_index)
            x = self.activation(x)
            x = self.dropout_conv(x)

        result = self.layer_out(x)
        result = self.dropout_output(result)
        return result
