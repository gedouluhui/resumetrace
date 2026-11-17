"""Exact typed evidence; values are hashed, never serialized as payloads."""
import hashlib
import math
import struct

import torch


_DTYPES = {torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32,
           torch.int64, torch.float16, torch.bfloat16, torch.float32,
           torch.float64, torch.complex64, torch.complex128}
_DTYPES.update(getattr(torch, name) for name in ('uint16', 'uint32', 'uint64')
               if hasattr(torch, name))
DTYPE_SIZES = {str(dtype): torch.empty((), dtype=dtype).element_size() for dtype in _DTYPES}
MAX_PATH = 4096


def _positive_int(value, name):
    if type(value) is not int:
        raise TypeError(name + '_type')
    if value <= 0:
        raise ValueError(name + '_limit')


def escape(segment):
    """JSON-pointer escaping applied after the key's type tag."""
    return segment.replace('~', '~0').replace('/', '~1')


def _children(value):
    if type(value) is dict:
        for key in value:
            if type(key) not in (str, int):
                raise TypeError('unsupported_key')
        for key in sorted(value, key=lambda k: (type(k).__name__, str(k))):
            yield ('s:' + escape(key) if type(key) is str else 'i:' + str(key)), value[key]
    else:
        tag = 'l:' if type(value) is list else 't:'
        for index, child in enumerate(value):
            yield tag + str(index), child


def fingerprint(value, *, max_tensor_bytes=67108864, max_leaves=20000):
    """Return a flat typed-path map. Container entries count toward max_leaves.

    Validate the entire graph and aggregate logical bytes before creating any
    contiguous tensor copy, including finite-value checks on tensor data.
    Repeated references are allowed; ancestor cycles are not.
    """
    _positive_int(max_tensor_bytes, 'tensor_bytes')
    _positive_int(max_leaves, 'leaves')
    nodes = []
    active = set()
    total_bytes = 0

    def validate(item, path, depth):
        nonlocal total_bytes
        if depth > 64:
            raise ValueError('depth_limit')
        if len(path) > MAX_PATH:
            raise ValueError('path_limit')
        if len(nodes) >= max_leaves:
            raise ValueError('leaf_limit')
        nodes.append((path, item))
        kind = type(item)
        if kind in (dict, list, tuple):
            if len(nodes) + len(item) > max_leaves:
                raise ValueError('leaf_limit')
            if id(item) in active:
                raise ValueError('cycle')
            active.add(id(item))
            for segment, child in _children(item):
                validate(child, path + '/' + segment, depth + 1)
            active.remove(id(item))
        elif kind in (torch.Tensor, torch.nn.Parameter):
            if item.is_nested or item.device.type != 'cpu' or item.layout != torch.strided or item.dtype not in _DTYPES:
                raise TypeError('unsupported_tensor')
            total_bytes += item.numel() * item.element_size()
            if total_bytes > max_tensor_bytes:
                raise ValueError('tensor_byte_limit')
        elif kind not in (type(None), bool, int, float, str):
            raise TypeError('unsupported_value')
        elif kind is float and not math.isfinite(item):
            raise ValueError('nonfinite_value')

    validate(value, '$', 0)
    result = {}
    for path, item in nodes:
        kind = type(item)
        if kind in (dict, list, tuple):
            result[path] = {'type': kind.__name__, 'length': len(item)}
            continue
        if kind in (torch.Tensor, torch.nn.Parameter):
            tensor = item.detach()
            if (tensor.is_floating_point() or tensor.is_complex()) and not torch.isfinite(tensor).all().item():
                raise ValueError('nonfinite_tensor')
            logical = tensor.resolve_conj().resolve_neg().contiguous().reshape(-1)
            payload = logical.view(torch.uint8).numpy().tobytes()
            result[path] = {'type': 'tensor', 'dtype': str(tensor.dtype),
                            'shape': list(tensor.shape), 'bytes': len(payload),
                            'sha256': hashlib.sha256(payload).hexdigest()}
            continue
        if kind is str:
            payload = item.encode('utf-8', errors='surrogatepass')
        elif kind is float:
            payload = struct.pack('!d', item)
        elif kind is int:
            payload = item.to_bytes((item.bit_length() + 8) // 8, 'big', signed=True)
        elif kind is bool:
            payload = b'1' if item else b'0'
        else:
            payload = b''
        result[path] = {'type': 'none' if item is None else kind.__name__,
                        'sha256': hashlib.sha256(payload).hexdigest()}
    return result


def compare_fingerprints(expected: dict, actual: dict) -> dict:
    """Compare every path exactly; cap details, never the mismatch count."""
    differences = []
    count = 0
    for path in sorted(expected.keys() | actual.keys()):
        if path not in expected or path not in actual or expected[path] != actual[path]:
            count += 1
            if len(differences) < 50:
                differences.append({'path': path, 'expected': expected.get(path),
                                    'actual': actual.get(path)})
    return {'matched': count == 0, 'differing_path_count': count,
            'differences': differences}
