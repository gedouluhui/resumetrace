"""One-line opt-in around the existing real SGD training example.

From the source checkout after installing the package:
  python -m resumetrace audit examples.guarded_training:make_session --steps 3 --split 1
  python -m resumetrace audit examples.guarded_training:make_session --config '{"fault":"save"}' --steps 3 --split 1

Faults are explicit educational callback mutations, not auditor verdict flags.
The underlying training loop and checkpoint format remain unchanged.
"""
import torch

from resumetrace import TorchSession
from examples.custom_training import RegressionSession


class EducationalSession(RegressionSession):
    def __init__(self, fault):
        super().__init__()
        self.fault = fault

    def observe(self):
        if self.fault == 'observer':
            with torch.no_grad():
                self.model.weight.zero_()
        return super().observe()

    def save(self, directory):
        super().save(directory)
        if self.fault == 'save':
            with torch.no_grad():
                self.model.weight.add_(1)


def make_session(config):
    if type(config) is not dict or set(config) - {'fault', 'gradient_policy'}:
        raise ValueError('guarded example accepts fault and gradient_policy')
    fault = config.get('fault', 'none')
    if type(fault) is not str or fault not in ('none', 'observer', 'save'):
        raise ValueError('unknown example fault')
    session = RegressionSession() if fault == 'none' else EducationalSession(fault)
    return TorchSession(session, gradient_policy=config.get('gradient_policy', 'ignore'))
