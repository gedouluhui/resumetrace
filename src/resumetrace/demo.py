"""A tiny, actual CPU training job with deliberately selectable checkpoint bugs.

Only this demonstration's own freshly written checkpoint is loaded. User
checkpoints are never automatically opened by the command-line interface.
"""
from __future__ import annotations

import copy
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn

FAULTS = ('none', 'optimizer', 'rng', 'cursor', 'scheduler', 'process-cache')
# Intentionally process-local: the named demo fault must not survive interpreter exit.
_PROCESS_CACHE: dict = {}


def _rng_state() -> dict:
    numpy_state = np.random.get_state()
    return {
        'torch': torch.get_rng_state(),
        'python': random.getstate(),
        'numpy': {
            'algorithm': numpy_state[0],
            'keys': numpy_state[1].tolist(),
            'position': numpy_state[2],
            'has_gauss': numpy_state[3],
            'cached_gaussian': numpy_state[4],
        },
    }


def _restore_rng(state: dict) -> None:
    torch.set_rng_state(state['torch'])
    random.setstate(state['python'])
    value = state['numpy']
    np.random.set_state((value['algorithm'], np.array(value['keys'], dtype=np.uint32),
                         value['position'], value['has_gauss'], value['cached_gaussian']))


class DemoSession:
    """Eight steps take no dataset downloads, GPU, network, or API credentials."""

    def __init__(self, fault: str):
        self.fault = fault
        self.model = nn.Sequential(nn.Linear(4, 12), nn.Tanh(), nn.Dropout(0.35), nn.Linear(12, 2))
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.025)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=2, gamma=0.7)
        source = torch.Generator().manual_seed(2026)
        self.features = torch.randn(41, 4, generator=source)
        self.targets = ((self.features[:, 0] + 0.4 * self.features[:, 1]) > 0).long()
        self.generator = torch.Generator().manual_seed(83)
        self.order = torch.randperm(len(self.features), generator=self.generator)
        self.position = 0
        self.last_ids = torch.empty(0, dtype=torch.int64)
        self.iteration = 0

    def _next_batch(self):
        # The harness never supplies or pre-caches these batches. A new process
        # re-enters this exact pipeline using only state restored by load().
        if self.position >= len(self.order):
            self.order = torch.randperm(len(self.features), generator=self.generator)
            self.position = 0
        end = min(self.position + 5, len(self.order))
        self.last_ids = self.order[self.position:end].clone()
        self.position = end
        return self.features[self.last_ids], self.targets[self.last_ids]

    def step(self) -> dict[str, float]:
        features, targets = self._next_batch()
        self.optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(self.model(features), targets)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.iteration += 1
        return {'loss': float(loss.detach()), 'step': self.iteration,
                'learning_rate': float(self.optimizer.param_groups[0]['lr'])}

    def observe(self) -> dict:
        # state_dict() is shallow; consumers fingerprint immediately. save()
        # serializes synchronously, rather than retaining these live references.
        return {
            'model': dict(self.model.state_dict()),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'data': {'position': self.position, 'order': self.order,
                     'generator': self.generator.get_state(), 'last_ids': self.last_ids},
            'training': self.model.training,
            'iteration': self.iteration,
        }

    def save(self, directory: Path) -> None:
        state = self.observe()
        state['rng'] = _rng_state()
        missing = {'optimizer': 'optimizer', 'rng': 'rng', 'cursor': 'data', 'scheduler': 'scheduler'}
        if self.fault in missing:
            state.pop(missing[self.fault])
        if self.fault == 'process-cache':
            _PROCESS_CACHE['optimizer'] = copy.deepcopy(state.pop('optimizer'))
        torch.save(state, directory / 'checkpoint.pt')

    def load(self, directory: Path) -> None:
        # Never fall back to weights_only=False on an error.
        state = torch.load(directory / 'checkpoint.pt', weights_only=True, map_location='cpu')
        self.model.load_state_dict(state['model'])
        self.model.train(state['training'])
        self.iteration = state['iteration']
        # Construct schedulers before optimizer load, then restore scheduler state
        # first so optimizer-restored LR is not overwritten by initialization.
        if 'scheduler' in state:
            self.scheduler.load_state_dict(state['scheduler'])
        if 'optimizer' in state:
            self.optimizer.load_state_dict(state['optimizer'])
        elif self.fault == 'process-cache' and 'optimizer' in _PROCESS_CACHE:
            self.optimizer.load_state_dict(copy.deepcopy(_PROCESS_CACHE['optimizer']))
        if 'data' in state:
            data = state['data']
            self.position = data['position']
            self.order = data['order']
            self.last_ids = data['last_ids']
            self.generator.set_state(data['generator'])
        if 'rng' in state:
            _restore_rng(state['rng'])


def make_session(config: dict) -> DemoSession:
    """Importable factory used by each fresh interpreter."""
    if type(config) is not dict or set(config) - {'fault'}:
        raise ValueError('demo config accepts only fault')
    fault = config.get('fault', 'none')
    if type(fault) is not str or fault not in FAULTS:
        raise ValueError('unsupported demo fault')
    return DemoSession(fault)
