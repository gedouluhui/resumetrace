"""Actual update/save/load callbacks for adapter integration tests."""
from pathlib import Path

import torch

from resumetrace.demo import DemoSession
from resumetrace.torch_observer import TorchSession


class GuardedFixture(DemoSession):
    def __init__(self, fault='none'):
        super().__init__('none')
        self.inspection_fault = fault
        if fault == 'optimizer-order':
            self.model = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False), torch.nn.Tanh(),
                                              torch.nn.Linear(4, 4, bias=False), torch.nn.Linear(4, 2))
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=.025)
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=2, gamma=.7)

    def observe(self):
        if self.inspection_fault == 'observer':
            with torch.no_grad():
                next(self.model.parameters()).zero_()
        # Native model/optimizer collection must not depend on these user keys.
        return {'scheduler': self.scheduler.state_dict(), 'iteration': self.iteration,
                'position': self.position, 'order': self.order,
                'generator': self.generator.get_state(), 'last_ids': self.last_ids}

    def save(self, directory):
        # DemoSession.save calls observe(); use its exact ordinary implementation
        # without importing the auditor into the checkpoint logic.
        from resumetrace.demo import _rng_state
        state = DemoSession.observe(self)
        state['rng'] = _rng_state()
        torch.save(state, directory / 'checkpoint.pt')
        if self.inspection_fault == 'save':
            with torch.no_grad():
                next(self.model.parameters()).add_(1)

    def load(self, directory):
        if self.inspection_fault == 'optimizer-order':
            params = self.optimizer.param_groups[0]['params']
            params[0], params[1] = params[1], params[0]
        super().load(directory)


def make_session(config):
    fault = config.get('fault', 'none')
    if fault in ('factory-late', 'generic-late'):
        marker = Path(config['marker'])
        seen = marker.exists()
        marker.write_text('seen', encoding='utf-8')
        if fault == 'factory-late' and seen:
            raise RuntimeError('late_factory_failure')
        if fault == 'generic-late':
            session = GuardedFixture('none')
            return session if seen else TorchSession(session)
    session = GuardedFixture(fault)
    if config.get('generic', False):
        return session
    return TorchSession(session, gradient_policy=config.get('gradient_policy', 'ignore'))
