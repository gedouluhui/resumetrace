"""Global-RNG omissions with identical numeric metrics and declared user state."""
from pathlib import Path
import random

import numpy as np
import torch


class Session:
    def __init__(self, config):
        self.omit = config.get('omit', 'none')
        self.colliding_keys = config.get('colliding_keys', False)
        self.value = 0

    def step(self):
        self.value += 1
        return {'loss': 1.0}

    def observe(self):
        state = {'counter': self.value}
        if self.colliding_keys:
            state.update(session={'counter': self.value}, global_rng={'counter': self.value})
        return state

    def save(self, directory: Path):
        n = np.random.get_state()
        state = {'counter': self.value, 'python': random.getstate(), 'torch': torch.get_rng_state(),
                 'numpy': (n[0], n[1].tolist(), n[2], n[3], n[4])}
        state.pop(self.omit, None)
        torch.save(state, directory / 'state.pt')

    def load(self, directory: Path):
        state = torch.load(directory / 'state.pt', map_location='cpu', weights_only=True)
        self.value = state['counter']
        if 'python' in state:
            random.setstate(state['python'])
        if 'torch' in state:
            torch.set_rng_state(state['torch'])
        if 'numpy' in state:
            n = state['numpy']
            np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))


def make(config):
    return Session(config)
