"""nn_utils.py -- vendored from ms-pred, imports fixed for speqtro.

DGL 2.x compatibility:
  - `dgl.backend.pytorch.pad_packed_tensor` and `pack_padded_tensor` are
    replaced with pure-torch implementations defined in this module.
  - `torch_scatter` is used as in the original.
"""
import copy
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
import dgl
from packaging.version import Version

if Version(torch.__version__) > Version('2.0.0'):
    _TORCH_SP_SUPPORT = True  # use torch built-in sparse
else:
    try:
        import torch_sparse
        _TORCH_SP_SUPPORT = False  # use torch_sparse package
    except Exception:
        raise ModuleNotFoundError("Please either install torch_sparse or upgrade to a PyTorch version that supports "
                                  "sparse-sparse matrix multiply")

from . import dgl_modules as dgl_mods


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_lr_scheduler(
    optimizer, lr_decay_rate: float, decay_steps: int = 5000, warmup: int = 1000
):
    def lr_lambda(step):
        if step >= warmup:
            step = step - warmup
            rate = lr_decay_rate ** (step // decay_steps)
        else:
            rate = 1 - math.exp(-step / warmup)
        return rate

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return scheduler


class MoleculeGNN(nn.Module):
    """MoleculeGNN Module"""

    def __init__(
        self,
        hidden_size: int,
        num_step_message_passing: int = 4,
        gnn_node_feats: int = 74,
        gnn_edge_feats: int = 4,
        mpnn_type: str = "GGNN",
        node_feat_symbol="h",
        set_transform_layers: int = 2,
        dropout: float = 0,
        **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.gnn_edge_feats = gnn_edge_feats
        self.gnn_node_feats = gnn_node_feats
        self.node_feat_symbol = node_feat_symbol
        self.dropout = dropout

        self.mpnn_type = mpnn_type
        self.hidden_size = hidden_size
        self.num_step_message_passing = num_step_message_passing
        self.input_project = nn.Linear(self.gnn_node_feats, self.hidden_size)

        if self.mpnn_type == "GGNN":
            self.gnn = GGNN(
                hidden_size=self.hidden_size,
                edge_feats=self.gnn_edge_feats,
                node_feats=self.gnn_node_feats,
                num_step_message_passing=num_step_message_passing,
            )
        elif self.mpnn_type == "PNA":
            self.gnn = PNA(
                hidden_size=self.hidden_size,
                edge_feats=self.gnn_edge_feats,
                node_feats=self.gnn_node_feats,
                num_step_message_passing=num_step_message_passing,
                dropout=self.dropout,
            )
        elif self.mpnn_type == "GINE":
            self.gnn = GINE(
                hidden_size=self.hidden_size,
                edge_feats=self.gnn_edge_feats,
                node_feats=self.gnn_node_feats,
                num_step_message_passing=num_step_message_passing,
                dropout=self.dropout,
            )
        else:
            raise ValueError()

        self.set_transformer = SetTransformerEncoder(
            d_model=self.hidden_size,
            n_heads=4,
            d_head=self.hidden_size // 4,
            d_ff=hidden_size,
            n_layers=set_transform_layers,
        )

    def forward(self, g):
        with g.local_scope():
            ndata = g.ndata[self.node_feat_symbol]
            edata = g.edata["e"]
            h_init = self.input_project(ndata)
            g.ndata.update({"_h": h_init})
            g.edata.update({"_e": edata})

            if self.mpnn_type in ("GGNN", "PNA", "GINE"):
                output = self.gnn(g, "_h", "_e")
            else:
                raise NotImplementedError()

        output = self.set_transformer(g, output)
        return output


class GINE(nn.Module):
    def __init__(self, hidden_size=64, edge_feats=4, num_step_message_passing=4, dropout=0, **kwargs):
        super().__init__()
        self.edge_transform = nn.Linear(edge_feats, hidden_size)

        self.layers = []
        for i in range(num_step_message_passing):
            apply_fn = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
            )
            temp_layer = dgl_mods.GINEConv(apply_func=apply_fn, init_eps=0)
            self.layers.append(temp_layer)

        self.layers = nn.ModuleList(self.layers)
        self.bnorms = get_clones(nn.BatchNorm1d(hidden_size), num_step_message_passing)
        self.dropouts = get_clones(nn.Dropout(dropout), num_step_message_passing)

    def forward(self, graph, nfeat_name="_h", efeat_name="_e"):
        node_feat, edge_feat = graph.ndata[nfeat_name], graph.edata[efeat_name]
        edge_feat = self.edge_transform(edge_feat)

        for dropout, layer, norm in zip(self.dropouts, self.layers, self.bnorms):
            layer_out = layer(graph, node_feat, edge_feat)
            node_feat = F.relu(dropout(norm(layer_out))) + node_feat

        return node_feat


class GGNN(nn.Module):
    def __init__(self, hidden_size=64, edge_feats=4, num_step_message_passing=4, **kwargs):
        super().__init__()
        self.model = dgl_mods.GatedGraphConv(
            in_feats=hidden_size,
            out_feats=hidden_size,
            n_steps=num_step_message_passing,
            n_etypes=edge_feats,
        )

    def forward(self, graph, nfeat_name="_h", efeat_name="_e"):
        if "e_ind" in graph.edata:
            etypes = graph.edata["e_ind"]
        else:
            etypes = graph.edata[efeat_name].argmax(1)
        return self.model(graph, graph.ndata[nfeat_name], etypes=etypes)


class PNA(nn.Module):
    def __init__(self, hidden_size=64, edge_feats=4, num_step_message_passing=4, dropout=0, **kwargs):
        super().__init__()
        self.layer = dgl_mods.PNAConv(
            in_size=hidden_size,
            out_size=hidden_size,
            aggregators=["mean", "max", "min", "std", "var", "sum"],
            scalers=["identity", "amplification", "attenuation"],
            delta=2.5,
            dropout=dropout,
        )
        self.layers = get_clones(self.layer, num_step_message_passing)
        self.bnorms = get_clones(nn.BatchNorm1d(hidden_size), num_step_message_passing)

    def forward(self, graph, nfeat_name="_h", efeat_name="_e"):
        node_feat, edge_feat = graph.ndata[nfeat_name], graph.edata[efeat_name]
        for layer, norm in zip(self.layers, self.bnorms):
            node_feat = F.relu(norm(layer(graph, node_feat, edge_feat))) + node_feat
        return node_feat


class MLPBlocks(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float,
        num_layers: int,
        output_size: int = None,
        use_residuals: bool = False,
        use_batchnorm: bool = False,
    ):
        super().__init__()
        self.activation = nn.ReLU()
        self.dropout_layer = nn.Dropout(p=dropout)
        self.input_layer = nn.Linear(input_size, hidden_size)
        middle_layer = nn.Linear(hidden_size, hidden_size)
        self.layers = get_clones(middle_layer, num_layers - 1)

        self.output_layer = None
        self.output_size = output_size
        if self.output_size is not None:
            self.output_layer = nn.Linear(hidden_size, self.output_size)

        self.use_residuals = use_residuals
        self.use_batchnorm = use_batchnorm
        if self.use_batchnorm:
            self.bn_input = nn.BatchNorm1d(hidden_size)
            bn = nn.BatchNorm1d(hidden_size)
            self.bn_mids = get_clones(bn, num_layers - 1)

    def safe_apply_bn(self, x, bn):
        temp_shape = x.shape
        if len(x.shape) == 2:
            return bn(x)
        elif len(x.shape) == 3:
            return bn(x.transpose(-1, -2)).transpose(-1, -2)
        else:
            raise NotImplementedError()

    def forward(self, x):
        output = x
        output = self.input_layer(x)
        output = self.activation(output)
        output = self.dropout_layer(output)

        if self.use_batchnorm:
            output = self.safe_apply_bn(output, self.bn_input)

        old_op = output
        for layer_index, layer in enumerate(self.layers):
            output = layer(output)
            output = self.activation(output)
            output = self.dropout_layer(output)

            if self.use_batchnorm:
                output = self.safe_apply_bn(output, self.bn_mids[layer_index])

            if self.use_residuals:
                output += old_op
                old_op = output

        if self.output_layer is not None:
            output = self.output_layer(output)

        return output


# -- Set Transformer (DGL) ----------------------------------------------------
# Pure-torch replacements for dgl.backend.pytorch helpers.
# These replace dgl_F.pad_packed_tensor and dgl_F.pack_padded_tensor which
# may not exist in DGL 2.x.

def pad_packed_tensor(input, lengths, value):
    """pad_packed_tensor -- pure-torch replacement for dgl_F.pad_packed_tensor."""
    old_shape = input.shape
    device = input.device
    if not isinstance(lengths, torch.Tensor):
        lengths = torch.tensor(lengths, dtype=torch.long, device=device)
    else:
        lengths = lengths.to(device)
    max_len = (lengths.max()).item()

    batch_size = len(lengths)
    x = input.new(batch_size * max_len, *old_shape[1:])
    x.fill_(value)

    index = torch.ones(len(input), dtype=torch.int64, device=device)
    row_shifts = torch.cumsum(max_len - lengths, 0)
    row_shifts_expanded = row_shifts[:-1].repeat_interleave(lengths[1:])
    cumsum_inds = torch.cumsum(index, 0) - 1
    cumsum_inds[lengths[0]:] += row_shifts_expanded
    x[cumsum_inds] = input
    return x.view(batch_size, max_len, *old_shape[1:])


def pack_padded_tensor(input, lengths):
    """pack_padded_tensor -- pure-torch replacement for dgl_F.pack_padded_tensor."""
    device = input.device
    if not isinstance(lengths, torch.Tensor):
        lengths = torch.tensor(lengths, dtype=torch.long, device=device)
    else:
        lengths = lengths.to(device)

    batch_size = len(lengths)
    packed_tensors = []
    for i in range(batch_size):
        packed_tensors.append(input[i, :lengths[i].item(), :])
    packed_tensors = torch.cat(packed_tensors)
    return packed_tensors


class MultiHeadAttention(nn.Module):
    r"""Multi-Head Attention block for Set Transformer."""

    def __init__(self, d_model, num_heads, d_head, d_ff, dropouth=0.0, dropouta=0.0):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_head
        self.d_ff = d_ff
        self.proj_q = nn.Linear(d_model, num_heads * d_head, bias=False)
        self.proj_k = nn.Linear(d_model, num_heads * d_head, bias=False)
        self.proj_v = nn.Linear(d_model, num_heads * d_head, bias=False)
        self.proj_o = nn.Linear(num_heads * d_head, d_model, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropouth),
            nn.Linear(d_ff, d_model),
        )
        self.droph = nn.Dropout(dropouth)
        self.dropa = nn.Dropout(dropouta)
        self.norm_in = nn.LayerNorm(d_model)
        self.norm_inter = nn.LayerNorm(d_model)
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def self_attention(self, x, mem, lengths_x, lengths_mem):
        batch_size = len(lengths_x)
        max_len_x = max(lengths_x)
        max_len_mem = max(lengths_mem)
        device = x.device

        lengths_x = lengths_x.clone().detach().long().to(device)
        lengths_mem = lengths_mem.clone().detach().long().to(device)

        queries = self.proj_q(x).view(-1, self.num_heads, self.d_head)
        keys = self.proj_k(mem).view(-1, self.num_heads, self.d_head)
        values = self.proj_v(mem).view(-1, self.num_heads, self.d_head)

        # DGL 2.x compat: use local pad/pack helpers instead of dgl_F
        queries = pad_packed_tensor(queries, lengths_x, 0)
        keys = pad_packed_tensor(keys, lengths_mem, 0)
        values = pad_packed_tensor(values, lengths_mem, 0)

        e = torch.einsum("bxhd,byhd->bhxy", queries, keys)
        e = e / np.sqrt(self.d_head)

        mask = _gen_mask(lengths_x, lengths_mem, max_len_x, max_len_mem)
        e = e.masked_fill(mask == 0, -float("inf"))

        alpha = torch.softmax(e, dim=-1)
        alpha = alpha.masked_fill(mask == 0, 0.0)

        out = torch.einsum("bhxy,byhd->bxhd", alpha, values)
        out = self.proj_o(
            out.contiguous().view(batch_size, max_len_x, self.num_heads * self.d_head)
        )
        # DGL 2.x compat: use local pack helper
        out = pack_padded_tensor(out, lengths_x)
        return out

    def forward(self, x, mem, lengths_x, lengths_mem):
        x = x + self.self_attention(self.norm_in(x), mem, lengths_x, lengths_mem)
        x = x + self.ffn(self.norm_inter(x))
        return x


class SetAttentionBlock(nn.Module):
    r"""SAB block from Set-Transformer."""

    def __init__(self, d_model, num_heads, d_head, d_ff, dropouth=0.0, dropouta=0.0):
        super(SetAttentionBlock, self).__init__()
        self.mha = MultiHeadAttention(
            d_model, num_heads, d_head, d_ff, dropouth=dropouth, dropouta=dropouta
        )

    def forward(self, feat, lengths):
        return self.mha(feat, feat, lengths, lengths)


class SetTransformerEncoder(nn.Module):
    r"""Set Transformer Encoder."""

    def __init__(
        self, d_model, n_heads, d_head, d_ff, n_layers=1,
        block_type="sab", m=None, dropouth=0.0, dropouta=0.0,
    ):
        super(SetTransformerEncoder, self).__init__()
        self.n_layers = n_layers
        self.block_type = block_type
        self.m = m
        layers = []
        if block_type == "isab" and m is None:
            raise KeyError("The number of inducing points is not specified in ISAB block.")

        for _ in range(n_layers):
            if block_type == "sab":
                layers.append(
                    SetAttentionBlock(
                        d_model, n_heads, d_head, d_ff,
                        dropouth=dropouth, dropouta=dropouta,
                    )
                )
            elif block_type == "isab":
                raise NotImplementedError()
            else:
                raise KeyError("Unrecognized block type {}: we only support sab/isab")

        self.layers = nn.ModuleList(layers)

    def forward(self, graph, feat):
        lengths = graph.batch_num_nodes()
        for layer in self.layers:
            feat = layer(feat, lengths)
        return feat


def _gen_mask(lengths_x, lengths_y, max_len_x, max_len_y):
    """Generate binary mask array for given x and y input pairs."""
    device = lengths_x.device
    x_mask = torch.arange(max_len_x, device=device).unsqueeze(0) < lengths_x.unsqueeze(1)
    y_mask = torch.arange(max_len_y, device=device).unsqueeze(0) < lengths_y.unsqueeze(1)
    mask = (x_mask.unsqueeze(-1) & y_mask.unsqueeze(-2)).unsqueeze(1)
    return mask


def random_walk_pe(g, k, eweight_name=None):
    """Random Walk Positional Encoding."""
    device = g.device
    N = g.num_nodes()
    M = g.num_edges()

    row, col = g.edges()

    if eweight_name is None:
        value = torch.ones(M, device=device)
    else:
        value = g.edata[eweight_name].squeeze().to(device)
    value_norm = torch_scatter.scatter(value, row, dim_size=N, reduce='sum')[row] + 1e-30
    value = value / value_norm

    if N <= 2_000:
        adj = torch.zeros((N, N), device=row.device)
        adj[row, col] = value
        loop_index = torch.arange(N, device=row.device)
    elif _TORCH_SP_SUPPORT:
        adj = torch.sparse_coo_tensor(indices=torch.stack((row, col)), values=value, size=(N, N))
    else:
        adj = torch_sparse.SparseTensor(row=row, col=col, value=value, sparse_sizes=(N, N))

    def get_pe(out: torch.Tensor) -> torch.Tensor:
        if not _TORCH_SP_SUPPORT and isinstance(out, torch_sparse.SparseTensor):
            return out.get_diag()
        elif _TORCH_SP_SUPPORT and out.is_sparse:
            out = out.coalesce()
            row_, col_ = out.indices()
            value_ = out.values()
            select = row_ == col_
            ret_val = torch.zeros(N, dtype=out.dtype, device=out.device)
            ret_val[row_[select]] = value_[select]
            return ret_val
        return out[loop_index, loop_index]

    out = adj
    pe_list = [get_pe(out)]
    for _ in range(k - 1):
        out = out @ adj
        pe_list.append(get_pe(out))

    pe = torch.stack(pe_list, dim=-1)
    return pe


def split_dgl_batch(batch: dgl.DGLGraph, max_dgl_edges, frag_hashes, rev_idx, frag_form_vecs):
    if batch.num_edges() > max_dgl_edges and batch.batch_size > 1:
        split = batch.batch_size // 2
        list_of_graphs = dgl.unbatch(batch)
        new_batch1 = split_dgl_batch(dgl.batch(list_of_graphs[:split]), max_dgl_edges,
                                     frag_hashes[:split], rev_idx[:split], frag_form_vecs[:split])
        new_batch2 = split_dgl_batch(dgl.batch(list_of_graphs[split:]), max_dgl_edges,
                                     frag_hashes[split:], rev_idx[split:], frag_form_vecs[split:])
        return new_batch1 + new_batch2
    else:
        return [(batch, frag_hashes, rev_idx, frag_form_vecs)]


def dict_to_device(data_dict, device):
    sent_dict = {}
    for key, value in data_dict.items():
        if torch.is_tensor(value):
            sent_dict[key] = value.to(device)
        else:
            sent_dict[key] = value
    return sent_dict
