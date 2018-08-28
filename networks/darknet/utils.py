
import numpy as np
import torch


class ShapeError(Exception):
    pass


def _load_into_tensor(tensor, weights, offset, size):
    tensor.data = torch.from_numpy(weights[offset:offset+size]).view_as(tensor.data)
    return offset + size


def load_weights(file, network):
    with open(file, 'rb') as fil:
        major, minor, revision = np.frombuffer(fil.read(12), dtype=np.uint32)
        if (10*major + minor) >= 2 and major < 1000 and minor < 1000:
            network.images_seen = int(np.frombuffer(fil.read(8), dtype=np.uint64)[0])
        else:
            network.images_seen = int(np.frombuffer(fil.read(4), dtype=np.uint32)[0])
        weights = np.fromfile(fil, dtype=np.float32)

    offset = 0
    for module in network.modules():
        if module.__class__.__name__ != 'ConvolutionalLayer':
            continue
        c = module.in_channels
        n = module.out_channels
        k = module.kernel_size
        if module.batch_normalize:
            offset = _load_into_tensor(module.batchnorm.bias,         weights, offset, n)
            offset = _load_into_tensor(module.batchnorm.weight,       weights, offset, n)
            offset = _load_into_tensor(module.batchnorm.running_mean, weights, offset, n)
            offset = _load_into_tensor(module.batchnorm.running_var,  weights, offset, n)
        else:
            offset = _load_into_tensor(module.conv.bias, weights, offset, n)
        offset = _load_into_tensor(module.conv.weight, weights, offset, n * c * k**2)
    
    if offset != weights.size:
        print('Warning: offset != weights.size in load_weights.',
              'offset = {}, weights.size = {}'.format(offset, weights.size))
