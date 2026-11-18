"""Opt-in native PyTorch observation and bounded callback-purity evidence.

This is not a sandbox. Only the declared, synchronously observed domain is
checked; filesystem effects and unregistered Python state are not certified.
"""
from collections import OrderedDict
import random

import numpy as np
import torch

from .fingerprint import MAX_PATH, compare_fingerprints, fingerprint

REASONS = frozenset({'observer_state_changed', 'save_state_changed',
                     'torch_inspection_unsupported', 'torch_inspection_budget'})
POLICIES = ('ignore', 'capture', 'require_none')


class InspectionError(ValueError):
    """Safe, value-free evidence about an opt-in inspection failure."""

    def __init__(self, reason, paths=(), count=0):
        if reason not in REASONS:
            raise ValueError('unknown_inspection_reason')
        self.reason, self.paths, self.count = reason, tuple(paths), count
        super().__init__(reason)


def _positive(value):
    if type(value) is not int or value <= 0:
        raise ValueError('inspection_budget')
    return value


def _rng():
    numpy = np.random.get_state()
    return {'python': random.getstate(),
            'numpy': (numpy[0], torch.from_numpy(numpy[1]), *numpy[2:]),
            'torch': torch.get_rng_state()}


class TorchSession:
    """Wrap a trusted CPU SGD/AdamW training session without changing its updates.

    ``session.model`` and ``session.optimizer`` must be ordinary instance
    attributes. Optional ``session.observe()`` supplies application state such
    as scheduler, data cursor and private generators. Native registrations are
    read directly; custom model serialization hooks are never invoked.

    Gradient policy controls *resume comparison*. Callback guards always inspect
    existing gradients. The default ignores gradients left after an optimizer
    step and never clears them. Saving must preserve observed state in this
    opt-in strict mode, even if a mutation happens after writing a valid file.
    """

    def __init__(self, session, *, gradient_policy='ignore',
                 max_tensor_bytes=67108864, max_leaves=20000):
        if type(gradient_policy) is not str or gradient_policy not in POLICIES:
            raise ValueError('gradient_policy')
        if any(not callable(getattr(session, name, None)) for name in ('step', 'save', 'load')):
            raise TypeError('session_protocol')
        observer = getattr(session, 'observe', None)
        if observer is not None and not callable(observer):
            raise TypeError('session_observer')
        self.session = session
        self.gradient_policy = gradient_policy
        self.max_tensor_bytes = _positive(max_tensor_bytes)
        self.max_leaves = _positive(max_leaves)
        self._declared_state = observer is not None

    def declaration(self):
        return {'mode': 'torch', 'gradient_policy': self.gradient_policy,
                'declared_state': self._declared_state}

    def limit(self, *, max_tensor_bytes, max_leaves):
        """Apply the auditor's cap before loading or calling a user's observer."""
        self.max_tensor_bytes = min(self.max_tensor_bytes, _positive(max_tensor_bytes))
        self.max_leaves = min(self.max_leaves, _positive(max_leaves))

    def step(self):
        return self.session.step()

    def load(self, directory):
        return self.session.load(directory)

    def _hash(self, value):
        try:
            return fingerprint(value, max_tensor_bytes=self.max_tensor_bytes,
                               max_leaves=self.max_leaves)
        except (TypeError, ValueError, RuntimeError) as error:
            code = str(error)
            reason = ('torch_inspection_budget' if code in
                      {'tensor_byte_limit', 'leaf_limit', 'depth_limit', 'path_limit'}
                      else 'torch_inspection_unsupported')
            raise InspectionError(reason) from None

    def _core(self, *, gradients):
        # Avoid property/state_dict hooks that could hide or modify native state.
        attributes = object.__getattribute__(self.session, '__dict__')
        model, optimizer = attributes.get('model'), attributes.get('optimizer')
        if not isinstance(model, torch.nn.Module) or type(optimizer) not in (torch.optim.SGD, torch.optim.AdamW):
            raise InspectionError('torch_inspection_unsupported')
        parameters, buffers, persistence, modes = {}, {}, {}, {}
        names, modules, intervals = {}, set(), []
        registration_count = 0

        def register(tensor, name, parameter=False):
            if tensor is None:
                return
            if (type(tensor) not in (torch.Tensor, torch.nn.Parameter) or
                    tensor.device.type != 'cpu' or tensor.layout != torch.strided or
                    tensor.is_nested or tensor.is_quantized or tensor.is_conj() or tensor.is_neg()):
                raise InspectionError('torch_inspection_unsupported')
            # A sufficient non-overlap certificate: each successive stride spans
            # all earlier dimensions. Ambiguous layouts are rejected, not guessed.
            if tensor.numel():
                span = 1
                for stride, size in sorted((s, n) for s, n in zip(tensor.stride(), tensor.shape) if n > 1):
                    if stride < span:
                        raise InspectionError('torch_inspection_unsupported')
                    span += (size - 1) * stride
            if parameter:
                if type(tensor) is not torch.nn.Parameter or id(tensor) in names:
                    raise InspectionError('torch_inspection_unsupported')
                names[id(tensor)] = name
            if tensor.numel():
                storage = tensor.untyped_storage()
                intervals.append((storage.data_ptr(), storage.data_ptr() + storage.nbytes()))

        def walk(module, prefix='', depth=0):
            nonlocal registration_count
            if depth > 64 or id(module) in modules:
                raise InspectionError('torch_inspection_unsupported')
            if not isinstance(module, torch.nn.Module):
                raise InspectionError('torch_inspection_unsupported')
            modules.add(id(module))
            attrs = object.__getattribute__(module, '__dict__')
            groups = [attrs.get(key) for key in ('_parameters', '_buffers', '_modules')]
            if any(type(group) not in (dict, OrderedDict) for group in groups):
                raise InspectionError('torch_inspection_unsupported')
            registration_count += 1 + sum(len(group) for group in groups)
            if registration_count > self.max_leaves:
                raise InspectionError('torch_inspection_budget')
            if type(attrs.get('training')) is not bool:
                raise InspectionError('torch_inspection_unsupported')
            modes[prefix] = attrs['training']
            for group in groups:
                if any(type(key) is not str or len(key) > MAX_PATH or '.' in key for key in group):
                    raise InspectionError('torch_inspection_unsupported')
            for key, tensor in groups[0].items():
                name = prefix + '.' + key if prefix else key
                register(tensor, name, parameter=True)
                parameters[name] = tensor
            for key, tensor in groups[1].items():
                name = prefix + '.' + key if prefix else key
                register(tensor, name)
                buffers[name] = tensor
                persistence[name] = key not in attrs.get('_non_persistent_buffers_set', set())
            for key, child in groups[2].items():
                if child is not None:
                    walk(child, prefix + '.' + key if prefix else key, depth + 1)

        walk(model)
        if not names:
            raise InspectionError('torch_inspection_unsupported')
        state, groups, owned = {}, [], set()
        for group in optimizer.param_groups:
            group_names = []
            if type(group) is not dict or type(group.get('params')) is not list:
                raise InspectionError('torch_inspection_unsupported')
            for parameter in group['params']:
                identity = id(parameter)
                if identity not in names or identity in owned:
                    raise InspectionError('torch_inspection_unsupported')
                owned.add(identity)
                name = names[identity]
                group_names.append(name)
                state[name] = optimizer.state.get(parameter, {})
            groups.append({**{k: v for k, v in group.items() if k != 'params'}, 'params': group_names})
        if any(id(parameter) not in owned for parameter in optimizer.state):
            raise InspectionError('torch_inspection_unsupported')
        active = set()

        def check_state(value, depth=0):
            nonlocal registration_count
            registration_count += 1
            if depth > 64 or registration_count > self.max_leaves:
                raise InspectionError('torch_inspection_budget')
            if isinstance(value, torch.Tensor):
                register(value, 'optimizer_or_gradient')
            elif type(value) in (dict, list, tuple):
                if id(value) in active:
                    raise InspectionError('torch_inspection_unsupported')
                if len(value) > self.max_leaves - registration_count:
                    raise InspectionError('torch_inspection_budget')
                active.add(id(value))
                for child in (value.values() if type(value) is dict else value):
                    check_state(child, depth + 1)
                active.remove(id(value))

        gradient_state = {n: p.grad for n, p in parameters.items() if p is not None}
        check_state({'state': state, 'groups': groups, 'gradients': gradient_state})
        end = -1
        for start, stop in sorted(intervals):
            if start < end:
                raise InspectionError('torch_inspection_unsupported')
            end = max(end, stop)
        result = {'model': {'parameters': parameters, 'buffers': buffers,
                            'buffer_persistence': persistence, 'modes': modes,
                            'requires_grad': {n: p.requires_grad for n, p in parameters.items() if p is not None}},
                  'optimizer': {'kind': type(optimizer).__name__, 'state': state, 'param_groups': groups}}
        if self.gradient_policy == 'require_none' and any(p is not None and p.grad is not None for p in parameters.values()):
            raise InspectionError('torch_inspection_unsupported')
        if gradients:
            result['gradients'] = {n: p.grad for n, p in parameters.items() if p is not None}
        return result

    @staticmethod
    def _check(before, after, reason):
        difference = compare_fingerprints(before, after)
        if not difference['matched']:
            raise InspectionError(reason, [item['path'] for item in difference['differences']],
                                  difference['differing_path_count'])

    def observe(self):
        before = self._hash({'torch': self._core(gradients=True), 'rng': _rng()})
        observer = getattr(self.session, 'observe', None)
        if (observer is not None) != self._declared_state:
            raise InspectionError('torch_inspection_unsupported')
        declared = observer() if observer is not None else {}
        after = self._hash({'torch': self._core(gradients=True), 'rng': _rng()})
        self._check(before, after, 'observer_state_changed')
        if type(declared) is not dict:
            raise InspectionError('torch_inspection_unsupported')
        result = {'torch': self._core(gradients=self.gradient_policy == 'capture'), 'declared': declared}
        self._hash(result)  # Validate combined data under the same observation budget.
        return result

    def _save_observation(self):
        observed = self.observe()
        return self._hash({'observed': observed, 'guard': self._core(gradients=True), 'rng': _rng()})

    def save(self, directory):
        before = self._save_observation()
        self.session.save(directory)
        after = self._save_observation()
        self._check(before, after, 'save_state_changed')
