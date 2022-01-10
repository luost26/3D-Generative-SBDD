import torch
import torch.nn.functional as F
from torch.nn import Module, Sequential, ModuleList, Linear
from torch_geometric.nn import MessagePassing, radius_graph
from math import pi as PI

from ..common import GaussianSmearing, ShiftedSoftplus


class CFConv(MessagePassing):

    def __init__(self, in_channels, out_channels, num_filters, edge_channels, cutoff=10.0):
        super().__init__(aggr='add')
        self.lin1 = Linear(in_channels, num_filters, bias=False)
        self.lin2 = Linear(num_filters, out_channels)
        self.nn = Sequential(
            Linear(edge_channels, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )   # Network for generating filter weights
        self.cutoff = cutoff

    def forward(self, x, edge_index, edge_length, edge_attr):
        W = self.nn(edge_attr)

        if self.cutoff is not None:
            C = 0.5 * (torch.cos(edge_length * PI / self.cutoff) + 1.0)
            C = C * (edge_length <= self.cutoff) * (edge_length >= 0.0)     # Modification: cutoff
            W = W * C.view(-1, 1)

        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)
        return x

    def message(self, x_j, W):
        return x_j * W


class InteractionBlock(Module):

    def __init__(self, hidden_channels, num_gaussians, num_filters, cutoff):
        super(InteractionBlock, self).__init__()
        self.conv = CFConv(hidden_channels, hidden_channels, num_filters, num_gaussians, cutoff)
        self.act = ShiftedSoftplus()
        self.lin = Linear(hidden_channels, hidden_channels)

    def forward(self, x, edge_index, edge_length, edge_attr):
        x = self.conv(x, edge_index, edge_length, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x


class SchNetEncoder(Module):

    def __init__(self, hidden_channels=128, num_filters=128,
                num_interactions=6, edge_channels=64, cutoff=10.0):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=edge_channels)
        self.cutoff = cutoff

        self.interactions = ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(hidden_channels, edge_channels,
                                     num_filters, cutoff)
            self.interactions.append(block)

    @property
    def out_channels(self):
        return self.hidden_channels

    def forward(self, node_attr, pos, batch):
        edge_index = radius_graph(pos, self.cutoff, batch=batch, loop=False)
        edge_length = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1)
        edge_attr = self.distance_expansion(edge_length)
        h = node_attr
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_length, edge_attr)
        return h
