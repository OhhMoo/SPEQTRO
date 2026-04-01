"""dgl_modules.py -- vendored from ms-pred, imports fixed for speqtro.

DGL 2.x compatibility: expand_as_pair imported from dgl_compat shim.
"""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import dgl.function as fn

from ..dgl_compat import expand_as_pair
import dgl.nn as dgl_nn
import dgl

gcn_msg = fn.copy_u(u="h", out="m")
gcn_reduce = fn.sum(msg="m", out="h")


class GatedGraphConv(nn.Module):
    r"""Gated Graph Convolution layer from `Gated Graph Sequence
    Neural Networks <https://arxiv.org/pdf/1511.05493.pdf>`__
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

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, etypes=None):
        with graph.local_scope():
            assert graph.is_homogeneous, (
                "not a homogeneous graph; convert it with to_homogeneous "
                "and pass in the edge type as argument"
            )
            if self._n_etypes != 1:
                assert (
                    etypes.min() >= 0 and etypes.max() < self._n_etypes
                ), "edge type indices out of range [0, {})".format(self._n_etypes)

            zero_pad = feat.new_zeros((feat.shape[0], self._out_feats - feat.shape[1]))
            feat = torch.cat([feat, zero_pad], -1)

            for _ in range(self._n_steps):
                if self._n_etypes == 1 and etypes is None:
                    graph.ndata["h"] = self.linears[0](feat)
                    graph.update_all(fn.copy_u("h", "m"), fn.sum("m", "a"))
                    a = graph.ndata.pop("a")
                else:
                    graph.ndata["h"] = feat
                    for i in range(self._n_etypes):
                        eids = (
                            torch.nonzero(etypes == i, as_tuple=False)
                            .contiguous()
                            .view(-1)
                            .type(graph.idtype)
                        )
                        if len(eids) > 0:
                            graph.apply_edges(
                                lambda edges: {
                                    "W_e*h": self.linears[i](edges.src["h"])
                                },
                                eids,
                            )
                    graph.update_all(fn.copy_e("W_e*h", "m"), fn.sum("m", "a"))
                    a = graph.ndata.pop("a")
                feat = self.gru(a, feat)
            return feat


def aggregate_mean(h):
    return torch.mean(h, dim=1)


def aggregate_max(h):
    return torch.max(h, dim=1)[0]


def aggregate_min(h):
    return torch.min(h, dim=1)[0]


def aggregate_sum(h):
    return torch.sum(h, dim=1)


def aggregate_std(h):
    return torch.sqrt(aggregate_var(h) + 1e-30)


def aggregate_var(h):
    h_mean_squares = torch.mean(h * h, dim=1)
    h_mean = torch.mean(h, dim=1)
    var = torch.relu(h_mean_squares - h_mean * h_mean)
    return var


def _aggregate_moment(h, n):
    h_mean = torch.mean(h, dim=1, keepdim=True)
    h_n = torch.mean(torch.pow(h - h_mean, n), dim=1)
    rooted_h_n = torch.sign(h_n) * torch.pow(torch.abs(h_n) + 1e-30, 1.0 / n)
    return rooted_h_n


def aggregate_moment_3(h):
    return _aggregate_moment(h, n=3)


def aggregate_moment_4(h):
    return _aggregate_moment(h, n=4)


def aggregate_moment_5(h):
    return _aggregate_moment(h, n=5)


def scale_identity(h):
    return h


def scale_amplification(h, D, delta):
    return h * (np.log(D + 1) / delta)


def scale_attenuation(h, D, delta):
    return h * (delta / np.log(D + 1))


AGGREGATORS = {
    "mean": aggregate_mean,
    "sum": aggregate_sum,
    "max": aggregate_max,
    "min": aggregate_min,
    "std": aggregate_std,
    "var": aggregate_var,
    "moment3": aggregate_moment_3,
    "moment4": aggregate_moment_4,
    "moment5": aggregate_moment_5,
}
SCALERS = {
    "identity": scale_identity,
    "amplification": scale_amplification,
    "attenuation": scale_attenuation,
}


class PNAConvTower(nn.Module):
    def __init__(self, in_size, out_size, aggregators, scalers, delta, dropout=0.0, edge_feat_size=0):
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

    def reduce_func(self, nodes):
        msg = nodes.mailbox["msg"]
        degree = msg.size(1)
        h = torch.cat([AGGREGATORS[agg](msg) for agg in self.aggregators], dim=1)
        h = torch.cat(
            [
                SCALERS[scaler](h, D=degree, delta=self.delta)
                if scaler != "identity"
                else h
                for scaler in self.scalers
            ],
            dim=1,
        )
        return {"h_neigh": h}

    def message(self, edges):
        if self.edge_feat_size > 0:
            f = torch.cat([edges.src["h"], edges.dst["h"], edges.data["a"]], dim=-1)
        else:
            f = torch.cat([edges.src["h"], edges.dst["h"]], dim=-1)
        return {"msg": self.M(f)}

    def forward(self, graph, node_feat, edge_feat=None):
        snorm_n = torch.cat(
            [torch.ones(N, 1).to(node_feat) / N for N in graph.batch_num_nodes()], dim=0
        ).sqrt()
        with graph.local_scope():
            graph.ndata["h"] = node_feat
            if self.edge_feat_size > 0:
                assert edge_feat is not None, "Edge features must be provided."
                graph.edata["a"] = edge_feat

            graph.update_all(self.message, self.reduce_func)
            h = self.U(torch.cat([node_feat, graph.ndata["h_neigh"]], dim=-1))
            h = h * snorm_n
            return self.dropout(self.batchnorm(h))


class PNAConv(nn.Module):
    r"""Principal Neighbourhood Aggregation Layer."""

    def __init__(self, in_size, out_size, aggregators, scalers, delta,
                 dropout=0.0, num_towers=1, edge_feat_size=0, residual=True):
        super(PNAConv, self).__init__()

        self.in_size = in_size
        self.out_size = out_size
        assert in_size % num_towers == 0
        assert out_size % num_towers == 0
        self.tower_in_size = in_size // num_towers
        self.tower_out_size = out_size // num_towers
        self.edge_feat_size = edge_feat_size
        self.residual = residual
        if self.in_size != self.out_size:
            self.residual = False

        self.towers = nn.ModuleList(
            [
                PNAConvTower(
                    self.tower_in_size, self.tower_out_size,
                    aggregators, scalers, delta,
                    dropout=dropout, edge_feat_size=edge_feat_size,
                )
                for _ in range(num_towers)
            ]
        )
        self.mixing_layer = nn.Sequential(nn.Linear(out_size, out_size), nn.LeakyReLU())

    def forward(self, graph, node_feat, edge_feat=None):
        h_cat = torch.cat(
            [
                tower(
                    graph,
                    node_feat[:, ti * self.tower_in_size : (ti + 1) * self.tower_in_size],
                    edge_feat,
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

    def message(self, edges):
        return {"m": F.relu(edges.src["hn"] + edges.data["he"])}

    def forward(self, graph, node_feat, edge_feat):
        with graph.local_scope():
            feat_src, feat_dst = expand_as_pair(node_feat, graph)
            graph.srcdata["hn"] = feat_src
            graph.edata["he"] = edge_feat
            graph.update_all(self.message, fn.sum("m", "neigh"))
            rst = (1 + self.eps) * feat_dst + graph.dstdata["neigh"]
            if self.apply_func is not None:
                rst = self.apply_func(rst)
            return rst


class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, feature):
        with g.local_scope():
            g.ndata["h"] = feature
            g.update_all(gcn_msg, gcn_reduce)
            h = g.ndata["h"]
            return self.linear(h)


class HyperGNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_conv, dropout=0):
        super(HyperGNN, self).__init__()
        self.layer = GCNLayer(hidden_size, hidden_size)
        self.layer_out = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()
        self.dropout_conv = nn.Dropout(dropout)
        self.dropout_output = nn.Dropout(dropout)
        self.num_conv = num_conv

    def forward(self, g, features):
        for _ in range(self.num_conv):
            features = self.layer(g, features)
            features = self.activation(features)
            features = self.dropout_conv(features)

        result = self.layer_out(features)
        result = self.dropout_output(result)
        return result
