"""Run from the repository root:

python -m resumetrace audit examples.custom_training:make_session --steps 6 --split 3

This is trusted training code. Replace these four methods with your own code;
do not replace your production save/load with a "fixed" audit-only checkpoint.
"""
from pathlib import Path
import random

import numpy as np
import torch


class RegressionSession:
    def __init__(self):
        self.model = torch.nn.Linear(2, 1)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.03, momentum=0.9)
        self.inputs = torch.tensor([[1., 2.], [-1., 0.5], [0.3, -0.4]])
        self.targets = torch.tensor([[1.], [-0.5], [0.2]])
        self.cursor = 0

    def step(self):
        index = self.cursor % len(self.inputs)
        self.optimizer.zero_grad(set_to_none=True)
        loss = (self.model(self.inputs[index]) - self.targets[index]).square().mean()
        loss.backward()
        self.optimizer.step()
        self.cursor += 1
        return {'loss': loss.detach().item()}

    def observe(self):
        return {'model': dict(self.model.state_dict()), 'optimizer': self.optimizer.state_dict(),
                'cursor': self.cursor, 'training': self.model.training}

    def save(self, directory: Path):
        # NumPy's uint32 ndarray is encoded as built-in values for restricted load.
        state = np.random.get_state()
        checkpoint = self.observe()
        checkpoint.update(torch_rng=torch.get_rng_state(), python_rng=random.getstate(),
                          numpy_rng=(state[0], state[1].tolist(), state[2], state[3], state[4]))
        torch.save(checkpoint, directory / 'checkpoint.pt')

    def load(self, directory: Path):
        state = torch.load(directory / 'checkpoint.pt', weights_only=True, map_location='cpu')
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.model.train(state['training'])
        self.cursor = state['cursor']
        random.setstate(state['python_rng'])
        numpy_rng = state['numpy_rng']
        np.random.set_state((numpy_rng[0], np.array(numpy_rng[1], dtype=np.uint32),
                             numpy_rng[2], numpy_rng[3], numpy_rng[4]))
        torch.set_rng_state(state['torch_rng'])


def make_session(config):
    if config:
        raise ValueError('this example accepts an empty configuration')
    return RegressionSession()
