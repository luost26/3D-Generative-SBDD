import torch
import torch.nn.functional as F
from torch.nn import Module, Sequential, ModuleList, Linear, Conv1d
from torch_geometric.nn import radius_graph, knn_graph
from torch_scatter import scatter_sum, scatter_softmax
from math import pi as PI

from ..common import GaussianSmearing, ShiftedSoftplus


class AttentionInteractionBlock(Module):

    def __init__(self, hidden_channels, edge_channels, key_channels, num_heads=1):
        super().__init__()

        assert hidden_channels % num_heads == 0 
        assert key_channels % num_heads == 0

        self.hidden_channels = hidden_channels
        self.key_channels = key_channels
        self.num_heads = num_heads

        self.k_lin = Conv1d(hidden_channels, key_channels, 1, groups=num_heads, bias=False)
        self.q_lin = Conv1d(hidden_channels, key_channels, 1, groups=num_heads, bias=False)
        self.v_lin = Conv1d(hidden_channels, hidden_channels, 1, groups=num_heads, bias=False)

        self.weight_k_net = Sequential(
            Linear(edge_channels, key_channels//num_heads),
            ShiftedSoftplus(),
            Linear(key_channels//num_heads, key_channels//num_heads),
        )
        self.weight_k_lin = Linear(key_channels//num_heads, key_channels//num_heads)

        self.weight_v_net = Sequential(
            Linear(edge_channels, hidden_channels//num_heads),
            ShiftedSoftplus(),
            Linear(hidden_channels//num_heads, hidden_channels//num_heads),
        )
        self.weight_v_lin = Linear(hidden_channels//num_heads, hidden_channels//num_heads)

        self.centroid_lin = Linear(hidden_channels, hidden_channels)
        self.act = ShiftedSoftplus()
        self.out_transform = Linear(hidden_channels, hidden_channels)

    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x:  Node features, (N, H).
            edge_index: (2, E).
            edge_attr:  (E, H)
        """
        N = x.size(0)
        row, col = edge_index   # (E,) , (E,)

        # Project to multiple key, query and value spaces
        h_keys = self.k_lin(x.unsqueeze(-1)).view(N, self.num_heads, -1)    # (N, heads, K_per_head)  
        h_queries = self.q_lin(x.unsqueeze(-1)).view(N, self.num_heads, -1) # (N, heads, K_per_head)
        h_values = self.v_lin(x.unsqueeze(-1)).view(N, self.num_heads, -1)  # (N, heads, H_per_head)

        # Compute keys and queries
        W_k = self.weight_k_net(edge_attr)  # (E, K_per_head)
        keys_j = self.weight_k_lin(W_k.unsqueeze(1) * h_keys[col])  # (E, heads, K_per_head)
        queries_i = h_queries[row]    # (E, heads, K_per_head)

        # Compute attention weights (alphas)
        qk_ij = (queries_i * keys_j).sum(-1)  # (E, heads)
        alpha = scatter_softmax(qk_ij, row, dim=0)

        # Compose messages
        W_v = self.weight_v_net(edge_attr)  # (E, H_per_head)
        msg_j = self.weight_v_lin(W_v.unsqueeze(1) * h_values[col])  # (E, heads, H_per_head)
        msg_j = alpha.unsqueeze(-1) * msg_j   # (E, heads, H_per_head)

        # Aggregate messages
        aggr_msg = scatter_sum(msg_j, row, dim=0, dim_size=N).view(N, -1) # (N, heads*H_per_head)
        out = self.centroid_lin(x) + aggr_msg

        out = self.out_transform(self.act(out))
        return out


class CFTransformerEncoder(Module):
    
    def __init__(self, hidden_channels=256, edge_channels=64, key_channels=128, num_heads=4, num_interactions=6, k=32, cutoff=10.0):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.edge_channels = edge_channels
        self.key_channels = key_channels
        self.num_heads = num_heads
        self.num_interactions = num_interactions
        self.k = k
        self.cutoff = cutoff


        self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=edge_channels)
        self.interactions = ModuleList()
        for _ in range(num_interactions):
            block = AttentionInteractionBlock(
                hidden_channels=hidden_channels,
                edge_channels=edge_channels,
                key_channels=key_channels,
                num_heads=num_heads,
            )
            self.interactions.append(block)

    @property
    def out_channels(self):
        return self.hidden_channels

    def forward(self, node_attr, pos, batch):
        # edge_index = radius_graph(pos, self.cutoff, batch=batch, loop=False)
        edge_index = knn_graph(pos, k=self.k, batch=batch, flow='target_to_source')
        edge_length = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1)
        edge_attr = self.distance_expansion(edge_length)

        h = node_attr
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_attr)
        return h


if __name__ == '__main__':
    from torch_geometric.data import Data, Batch

    hidden_channels = 64
    edge_channels = 48
    key_channels = 32
    num_heads = 4

    data_list = []
    for num_nodes in [11, 13, 15]:
        data_list.append(Data(
            x = torch.randn([num_nodes, hidden_channels]),
            pos = torch.randn([num_nodes, 3]) * 2
        ))
    batch = Batch.from_data_list(data_list)

    model = CFTransformerEncoder(
        hidden_channels = hidden_channels,
        edge_channels = edge_channels,
        key_channels = key_channels,
        num_heads = num_heads,
    )
    out = model(batch.x, batch.pos, batch.batch)

    print(out)
    print(out.size())
