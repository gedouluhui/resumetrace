"""Real object and mutation controls for the optional PyTorch adapter."""
import copy
import importlib.util
import random
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from resumetrace.fingerprint import fingerprint

if importlib.util.find_spec('resumetrace.torch_observer') is not None:
    from resumetrace.torch_observer import InspectionError, TorchSession
else:
    InspectionError = TorchSession = None


class Pair(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.left = torch.nn.Parameter(torch.tensor([1., 2.]))
        self.right = torch.nn.Parameter(torch.tensor([3., 4.]))
        self.register_buffer('scratch', torch.tensor([5.]), persistent=False)


class Session:
    def __init__(self):
        self.model = Pair()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=.1, momentum=.9)
        self.counter = 0

    def observe(self):
        return {'counter': self.counter}

    def step(self):
        self.optimizer.zero_grad(set_to_none=True)
        loss = (self.model.left.square().sum() + self.model.right.sum())
        loss.backward()
        self.optimizer.step()
        self.counter += 1
        return {'loss': loss.item()}

    def save(self, directory):
        torch.save({'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()},
                   directory / 'state.pt')

    def load(self, directory):
        state = torch.load(directory / 'state.pt', weights_only=True)
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])


class TorchObserverTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(TorchSession, 'TorchSession is not implemented')

    def test_real_parameter_binding_not_serialized_ids(self):
        a, b = Session(), Session()
        a.step()
        b.model.load_state_dict(a.model.state_dict())
        b.counter = a.counter
        b.optimizer.param_groups[0]['params'].reverse()
        b.optimizer.load_state_dict(a.optimizer.state_dict())
        self.assertEqual(fingerprint(a.optimizer.state_dict()), fingerprint(b.optimizer.state_dict()))
        left, right = fingerprint(TorchSession(a).observe()), fingerprint(TorchSession(b).observe())
        changed = [p for p in left if left[p] != right[p]]
        self.assertTrue(any('/s:optimizer/s:state/s:left/' in p for p in changed), changed)
        self.assertTrue(any('/s:optimizer/s:param_groups/' in p for p in changed), changed)

    def test_normal_observation_does_not_change_state_or_rng(self):
        session = Session()
        before = fingerprint({'weights': dict(session.model.state_dict()),
                              'rng': torch.get_rng_state(), 'python': random.getstate(),
                              'numpy': torch.from_numpy(np.random.get_state()[1])})
        wrapped = TorchSession(session)
        observed = wrapped.observe()
        after = fingerprint({'weights': dict(session.model.state_dict()),
                             'rng': torch.get_rng_state(), 'python': random.getstate(),
                             'numpy': torch.from_numpy(np.random.get_state()[1])})
        self.assertEqual(before, after)
        self.assertEqual(len(session.optimizer.state), 0)
        self.assertIn('scratch', observed['torch']['model']['buffers'])
        self.assertEqual(wrapped.declaration(), {'mode': 'torch', 'gradient_policy': 'ignore',
                                                'declared_state': True})

    def test_saturating_observer_changes_weight_before_return(self):
        session = Session()
        def bad_observe():
            with torch.no_grad():
                session.model.left.zero_()
            return {'counter': session.counter}
        session.observe = bad_observe
        with self.assertRaises(InspectionError) as caught:
            TorchSession(session).observe()
        self.assertEqual(caught.exception.reason, 'observer_state_changed')
        self.assertTrue(any('left' in path for path in caught.exception.paths))

    def test_observer_rng_consumption_detected(self):
        session = Session()
        def bad_observe():
            torch.rand(1)
            return {}
        session.observe = bad_observe
        with self.assertRaises(InspectionError) as caught:
            TorchSession(session).observe()
        self.assertEqual(caught.exception.reason, 'observer_state_changed')
        self.assertTrue(any('/s:rng/' in path for path in caught.exception.paths))

    def test_valid_checkpoint_with_save_side_effect_has_evidence(self):
        session = Session()
        original = session.save
        def save(directory):
            original(directory)
            with torch.no_grad():
                session.model.left.add_(1)
        session.save = save
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(InspectionError) as caught:
                TorchSession(session).save(Path(directory))
            self.assertTrue((Path(directory) / 'state.pt').is_file())
        self.assertEqual(caught.exception.reason, 'save_state_changed')
        self.assertTrue(any('left' in path for path in caught.exception.paths))

    def test_save_declared_counter_change_is_detected(self):
        session = Session()
        original = session.save
        def save(directory):
            original(directory)
            session.counter += 1
        session.save = save
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(InspectionError) as caught:
                TorchSession(session).save(Path(directory))
        self.assertIn('$/s:observed/s:declared/s:counter', caught.exception.paths)

    def test_default_gradient_policy_does_not_clear_or_compare_gradients(self):
        a, b = Session(), Session()
        a.step()
        b.model.load_state_dict(a.model.state_dict())
        b.optimizer.load_state_dict(a.optimizer.state_dict())
        b.counter = a.counter
        gradient = a.model.left.grad.clone()
        self.assertEqual(fingerprint(TorchSession(a).observe()), fingerprint(TorchSession(b).observe()))
        self.assertTrue(torch.equal(a.model.left.grad, gradient))
        self.assertNotEqual(fingerprint(TorchSession(a, gradient_policy='capture').observe()),
                            fingerprint(TorchSession(b, gradient_policy='capture').observe()))
        with self.assertRaises(InspectionError):
            TorchSession(a, gradient_policy='require_none').observe()

    def test_gradient_mutation_guard_even_when_not_compared_on_resume(self):
        session = Session()
        session.step()
        def bad_observe():
            session.model.left.grad.zero_()
            return {}
        session.observe = bad_observe
        with self.assertRaises(InspectionError) as caught:
            TorchSession(session).observe()
        self.assertTrue(any('/s:gradients/' in path for path in caught.exception.paths))

    def test_modes_requires_grad_and_nonpersistent_buffers(self):
        session = Session()
        wrapped = TorchSession(session)
        original = fingerprint(wrapped.observe())
        session.model.eval()
        session.model.right.requires_grad_(False)
        session.model.scratch.add_(1)
        updated = fingerprint(wrapped.observe())
        changed = [p for p in original if original[p] != updated[p]]
        for token in ('modes', 'requires_grad', 'scratch'):
            self.assertTrue(any(token in p for p in changed), changed)

    def test_reads_registrations_not_user_serialization_hooks(self):
        session = Session()
        session.model.state_dict = lambda *a, **kw: {}
        before = fingerprint(TorchSession(session).observe())
        with torch.no_grad():
            session.model.left.add_(1)
        self.assertNotEqual(before, fingerprint(TorchSession(session).observe()))

    def test_alias_ownership_and_unsupported_optimizer(self):
        for fault in ('alias', 'foreign', 'optimizer'):
            session = Session()
            if fault == 'alias':
                session.model.register_buffer('aliased', session.model.left.detach())
            elif fault == 'foreign':
                session.optimizer.add_param_group({'params': [torch.nn.Parameter(torch.ones(2))]})
            else:
                session.optimizer = torch.optim.RMSprop(session.model.parameters())
            with self.subTest(fault=fault), self.assertRaises(InspectionError) as caught:
                TorchSession(session).observe()
            self.assertEqual(caught.exception.reason, 'torch_inspection_unsupported')

    def test_callback_cannot_alias_optimizer_state_to_parameter(self):
        for callback in ('observe', 'save'):
            session = Session()
            parameter = session.model.left
            session.optimizer.state[parameter]['momentum_buffer'] = parameter.detach().clone()
            original = getattr(session, callback)
            def mutate(*args):
                result = original(*args)
                session.optimizer.state[parameter]['momentum_buffer'] = parameter.detach()
                return result
            setattr(session, callback, mutate)
            with self.subTest(callback=callback), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(InspectionError) as caught:
                    wrapped = TorchSession(session)
                    wrapped.observe() if callback == 'observe' else wrapped.save(Path(directory))
                self.assertEqual(caught.exception.reason, 'torch_inspection_unsupported')

    def test_alias_negative_control_changes_real_sgd_update(self):
        values = []
        for alias in (False, True):
            parameter = torch.nn.Parameter(torch.ones(1))
            optimizer = torch.optim.SGD([parameter], lr=.1, momentum=.9)
            optimizer.state[parameter]['momentum_buffer'] = parameter.detach() if alias else parameter.detach().clone()
            parameter.sum().backward()
            optimizer.step()
            values.append(parameter.item())
        self.assertAlmostEqual(values[0], .81, places=6)
        self.assertAlmostEqual(values[1], 1.71, places=6)
        self.assertNotEqual(values[0], values[1])

    def test_internal_overlap_introduced_by_callback_is_rejected(self):
        for callback in ('observe', 'save'):
            session = Session()
            with torch.no_grad():
                session.model.left.fill_(1)
            original = getattr(session, callback)
            def mutate(*args):
                result = original(*args)
                session.model.left.data = session.model.left.detach()[:1].expand(2)
                return result
            setattr(session, callback, mutate)
            with self.subTest(callback=callback), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(InspectionError) as caught:
                    wrapped = TorchSession(session)
                    wrapped.observe() if callback == 'observe' else wrapped.save(Path(directory))
                self.assertEqual(caught.exception.reason, 'torch_inspection_unsupported')
            session.model.left.sum().backward()
            with self.assertRaises(RuntimeError):
                session.optimizer.step()

    def test_gradient_parameter_storage_alias_is_rejected(self):
        session = Session()
        session.model.left.grad = session.model.left.detach()
        with self.assertRaises(InspectionError) as caught:
            TorchSession(session).observe()
        self.assertEqual(caught.exception.reason, 'torch_inspection_unsupported')

    def test_missing_extra_observer_declared_and_forwarding(self):
        session = Session()
        session.observe = None
        wrapped = TorchSession(session)
        self.assertFalse(wrapped.declaration()['declared_state'])
        self.assertEqual(wrapped.observe()['declared'], {})
        self.assertEqual(wrapped.step()['loss'], 12.0)
        self.assertEqual(session.counter, 1)

    def test_budget_validation_and_audit_cap(self):
        with self.assertRaises(ValueError):
            TorchSession(Session(), gradient_policy='guess')
        with self.assertRaises(ValueError):
            TorchSession(Session(), max_tensor_bytes=True)
        wrapped = TorchSession(Session())
        wrapped.limit(max_tensor_bytes=1, max_leaves=20000)
        with self.assertRaises(InspectionError) as caught:
            wrapped.observe()
        self.assertEqual(caught.exception.reason, 'torch_inspection_budget')


if __name__ == '__main__':
    unittest.main()
