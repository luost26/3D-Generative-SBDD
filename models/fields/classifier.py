import torch
from torch.nn import Module, Linear, Sequential
from torch_geometric.nn import radius, knn
from torch_scatter import scatter_add
from math import pi as PI

from ..common import ShiftedSoftplus, GaussianSmearing


class SpatialClassifier(Module):

    def __init__(self, num_classes, num_indicators, in_channels, num_filters, k=32, cutoff=10.0):
        super().__init__()
        self.lin1 = Linear(in_channels, num_filters, bias=False)
        self.lin2 = Linear(num_filters, num_filters)
        self.nn = Sequential(
            Linear(num_filters, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )
        self.classifier = Sequential(
            Linear(num_filters, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_classes),
        )
        self.property_pred = Sequential(
            Linear(num_filters, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_indicators),
        )
        self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=num_filters)
        self.k = k
        self.cutoff = cutoff

    def forward(self, pos_query, pos_ctx, node_attr_ctx, batch_query, batch_ctx):
        """
        Args:
            pos_query:   (N_query, 3)
            pos_ctx:     (N_ctx, 3)
            node_attr_ctx:  (N_ctx, H)
            batch_query: (N_query, )
            batch_ctx:   (N_ctx, )
        Returns
            (N_query, num_classes)
        """

        # For each element in `y` (pos_query), all points in `x` (pos_ctx) within distance `r` (cutoff)
        # assign_idx = radius(
        #     x=pos_ctx,
        #     y=pos_query,
        #     r=self.cutoff,
        #     batch_x=batch_ctx,
        #     batch_y=batch_query,
        # )   # (query_i, ctx_j) pairs
        assign_idx = knn(
            x=pos_ctx,
            y=pos_query,
            k=self.k,
            batch_x=batch_ctx,
            batch_y=batch_query,
        )

        # Pairwise distances and contextual node features
        dist_ij = torch.norm(pos_query[assign_idx[0]] - pos_ctx[assign_idx[1]], p=2, dim=-1).view(-1, 1)  # (A, 1)
        node_attr_ctx_j = node_attr_ctx[assign_idx[1]]  # (A, H)

        # Messages
        W = self.nn(self.distance_expansion(dist_ij))  # (A, F)
        h = self.lin2(W * self.lin1(node_attr_ctx_j))    # (A, F)

        # Message annealing
        C = 0.5 * (torch.cos(dist_ij * PI / self.cutoff) + 1.0)  # (A, 1)
        C = C * (dist_ij <= self.cutoff) * (dist_ij >= 0.0)
        h = h * C.view(-1, 1)   # (A, 1)

        # Aggregate messages
        y = scatter_add(h, index=assign_idx[0], dim=0, dim_size=pos_query.size(0)) # (N_query, F)
        
        y_cls = self.classifier(y)  # (N_query, num_classes)
        y_ind = self.property_pred(y)   # (N_query, num_indicators)

        return y_cls, y_ind
