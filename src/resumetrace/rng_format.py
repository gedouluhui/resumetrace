"""Runtime format of the three default RNG states.

A fingerprint carries dtype/shape metadata for tensor leaves and container
lengths. Checking that metadata against the format this interpreter actually
produces rejects structurally self-consistent but invalid RNG evidence (for
example a float32 empty Torch state, or a Python state tuple of the wrong
length) instead of treating it as comparable training state.

The templates are read from the live runtime once. They describe only the
default global generators, which is exactly what the worker observes; they do
not constrain user-declared session state.
"""
import random

import numpy
import torch


def templates():
    python_state = random.getstate()
    numpy_state = numpy.random.get_state()
    # The worker carries the NumPy key as a tensor, so declare the format of
    # that tensor rather than the NumPy dtype name.
    numpy_key = torch.from_numpy(numpy_state[1])
    torch_state = torch.get_rng_state()
    return {
        'python_version': int(python_state[0]),
        'python_state_length': len(python_state[1]),
        'python_gauss': 'none' if python_state[2] is None else 'float',
        'numpy_key': str(numpy_state[0]),
        'numpy_dtype': str(numpy_key.dtype),
        'numpy_shape': [int(size) for size in numpy_key.shape],
        'torch_dtype': str(torch_state.dtype),
        'torch_shape': [int(size) for size in torch_state.shape],
    }


def _leaf(state, path):
    leaf = state.get(path)
    if type(leaf) is not dict:
        raise ValueError('invalid_observation_evidence')
    return leaf


def _expect_scalar(state, path, kind):
    leaf = _leaf(state, path)
    if leaf.get('type') != kind or set(leaf) != {'type', 'sha256'}:
        raise ValueError('invalid_observation_evidence')


def _expect_container(state, path, kind, length):
    leaf = _leaf(state, path)
    if leaf != {'type': kind, 'length': length}:
        raise ValueError('invalid_observation_evidence')


def _expect_tensor(state, path, dtype, shape):
    leaf = _leaf(state, path)
    if (leaf.get('type') != 'tensor' or leaf.get('dtype') != dtype
            or leaf.get('shape') != shape):
        raise ValueError('invalid_observation_evidence')


def check(state, declared):
    """Reject RNG evidence that does not match the declared runtime format."""
    base = '$/s:global_rng'
    _expect_scalar(state, base + '/s:python/t:0', 'int')
    _expect_container(state, base + '/s:python/t:1', 'tuple', declared['python_state_length'])
    for index in range(declared['python_state_length']):
        _expect_scalar(state, base + '/s:python/t:1/t:%d' % index, 'int')
    _expect_scalar(state, base + '/s:python/t:2', declared['python_gauss'])

    _expect_scalar(state, base + '/s:numpy/t:0', 'str')
    _expect_tensor(state, base + '/s:numpy/t:1', declared['numpy_dtype'],
                   declared['numpy_shape'])
    _expect_scalar(state, base + '/s:numpy/t:2', 'int')
    _expect_scalar(state, base + '/s:numpy/t:3', 'int')
    _expect_scalar(state, base + '/s:numpy/t:4', 'float')

    _expect_tensor(state, base + '/s:torch', declared['torch_dtype'],
                   declared['torch_shape'])
