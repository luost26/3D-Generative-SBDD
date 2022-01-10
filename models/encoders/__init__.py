from .schnet import SchNetEncoder
from .cftfm import CFTransformerEncoder


def get_encoder(config):
    if config.name == 'schnet':
        return SchNetEncoder(
            hidden_channels = config.hidden_channels,
            num_filters = config.num_filters,
            num_interactions = config.num_interactions,
            edge_channels = config.edge_channels,
            cutoff = config.cutoff,
        )
    elif config.name == 'cftfm':
        return CFTransformerEncoder(
            hidden_channels = config.hidden_channels,
            edge_channels = config.edge_channels,
            key_channels = config.key_channels,
            num_heads = config.num_heads,
            num_interactions = config.num_interactions,
            k = config.knn,
            cutoff = config.cutoff,
        )
    else:
        raise NotImplementedError('Unknown encoder: %s' % config.name)
