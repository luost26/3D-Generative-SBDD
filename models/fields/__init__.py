from .classifier import SpatialClassifier


def get_field(config, num_classes, num_indicators, in_channels):
    if config.name == 'classifier':
        return SpatialClassifier(
            num_classes = num_classes,
            num_indicators = num_indicators,
            in_channels = in_channels,
            num_filters = config.num_filters,
            k = config.knn,
            cutoff = config.cutoff,
        )
    else:
        raise NotImplementedError('Unknown field: %s' % config.name)
