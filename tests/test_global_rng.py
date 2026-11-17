"""Cold-process regressions for RNG drift invisible to session state and loss."""
import copy
import json
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch

from resumetrace import audit_resume, engine


FACTORY = 'tests.fixtures.rng_only:make'


class GlobalRngTests(unittest.TestCase):
    def audit(self, omit):
        return audit_resume(FACTORY, factory_config={'omit': omit}, steps=3, split=1)

    def assert_constant_loss(self, report):
        self.assertIsNone(report['error'])
        self.assertNotIn('MT19937', json.dumps(report))
        self.assertEqual(report['control'], {'status': 'PASS', 'steps_checked': 3})
        self.assertEqual([worker['role'] for worker in report['workers']],
                         ['reference', 'control', 'prefix', 'resume'])
        self.assertEqual([(item['phase'], item['step']) for item in report['comparisons']],
                         [('restore', 1), ('step', 2), ('step', 3)])
        self.assertIsNone(report['comparisons'][0]['reference_metrics'])
        self.assertIsNone(report['comparisons'][0]['resumed_metrics'])
        for comparison in report['comparisons'][1:]:
            self.assertEqual(comparison['reference_metrics'], {'loss': 1.0})
            self.assertEqual(comparison['resumed_metrics'], {'loss': 1.0})

    def assert_omitted_rng_drifts(self, omit):
        report = self.audit(omit)
        self.assertEqual(report['status'], 'DRIFT', report['error'])
        self.assert_constant_loss(report)
        first = report['first_divergence']
        self.assertEqual((first['phase'], first['step']), ('restore', 1))
        self.assertEqual(first['paths'],
                         [item['path'] for item in report['comparisons'][0]['differences']])
        prefix = '$/state/s:global_rng/s:' + omit
        for comparison in report['comparisons']:
            self.assertFalse(comparison['matched'])
            self.assertGreater(comparison['differing_path_count'], 0)
            self.assertTrue(comparison['differences'])
            for difference in comparison['differences']:
                path = difference['path']
                self.assertTrue(path == prefix or path.startswith(prefix + '/'), path)
                for side in ('expected', 'actual'):
                    self.assertEqual(set(difference[side]),
                                     {'type', 'dtype', 'shape', 'bytes', 'sha256'}
                                     if difference[side]['type'] == 'tensor'
                                     else {'type', 'sha256'})
                    self.assertRegex(difference[side]['sha256'], '^[0-9a-f]{64}$')

    def test_omitted_python_rng_drifts_with_constant_loss(self):
        self.assert_omitted_rng_drifts('python')

    def test_omitted_numpy_rng_drifts_with_constant_loss(self):
        self.assert_omitted_rng_drifts('numpy')

    def test_omitted_torch_rng_drifts_with_constant_loss(self):
        self.assert_omitted_rng_drifts('torch')

    def test_parent_rejects_missing_rng_evidence_from_real_worker(self):
        captured = []
        original = engine._validate_result
        def capture(result, job, pid):
            if not captured:
                captured.append((copy.deepcopy(result), copy.deepcopy(job), pid))
            return original(result, job, pid)
        with patch.object(engine, '_validate_result', side_effect=capture):
            report = self.audit('none')
        self.assertEqual(report['status'], 'PASS', report['error'])
        result, job, pid = captured[0]
        for missing in ('global_rng', 'global_rng/s:python', 'global_rng/s:numpy',
                        'global_rng/s:torch'):
            with self.subTest(missing=missing):
                corrupted = copy.deepcopy(result)
                state = corrupted['observations'][0]['state']
                prefix = '$/s:' + missing
                for path in list(state):
                    if path == prefix or path.startswith(prefix + '/'):
                        del state[path]
                # Keep container counts consistent: generic fingerprint validation
                # alone must not accept a structurally valid but incomplete trace.
                state[prefix.rpartition('/')[0]]['length'] -= 1
                with self.assertRaises(ValueError):
                    original(corrupted, job, pid)
        for path in ('$/s:session', '$/s:global_rng/s:python',
                     '$/s:global_rng/s:numpy', '$/s:global_rng/s:numpy/t:1',
                     '$/s:global_rng/s:torch'):
            with self.subTest(wrong_type=path):
                corrupted = copy.deepcopy(result)
                state = corrupted['observations'][0]['state']
                for child in list(state):
                    if child.startswith(path + '/'):
                        del state[child]
                state[path] = {'type': 'none', 'sha256': '0' * 64}
                with self.assertRaises(ValueError):
                    original(corrupted, job, pid)
        with self.subTest(missing='legacy_unwrapped_state'):
            corrupted = copy.deepcopy(result)
            prefix = '$/s:session'
            corrupted['observations'][0]['state'] = {
                '$' + path[len(prefix):]: leaf
                for path, leaf in corrupted['observations'][0]['state'].items()
                if path == prefix or path.startswith(prefix + '/')}
            with self.assertRaises(ValueError):
                original(corrupted, job, pid)

    def test_global_rng_evidence_shares_exact_byte_leaf_and_trace_budgets(self):
        # Count the typed wrapper, counter, native Python tuple, NumPy tuple
        # (whose keys are one tensor), and the default torch state tensor.
        python_leaves = 4 + len(random.getstate()[1])
        state_leaves = 2 + 2 + python_leaves + 6 + 1
        torch_state = torch.get_rng_state()
        tensor_bytes = (np.random.get_state()[1].nbytes
                        + torch_state.numel() * torch_state.element_size())
        # N=2 yields ten observations across A/B/C/D and six metric entries.
        limits = {'max_tensor_bytes': tensor_bytes, 'max_leaves': state_leaves,
                  'max_trace_leaves': 10 * state_leaves + 6}
        report = audit_resume(FACTORY, steps=2, split=1, **limits)
        self.assertEqual(report['status'], 'PASS', report['error'])
        self.assertEqual(report['coverage']['trace_leaves'], limits['max_trace_leaves'])
        for key, stage, reason, workers in (
                ('max_tensor_bytes', 'fingerprint', 'invalid_state', 1),
                ('max_leaves', 'fingerprint', 'invalid_state', 1),
                ('max_trace_leaves', 'result', 'trace_leaf_limit', 4)):
            with self.subTest(budget=key):
                reduced = {**limits, key: limits[key] - 1}
                report = audit_resume(FACTORY, steps=2, split=1, **reduced)
                self.assertEqual(report['status'], 'INVALID')
                self.assertEqual(report['error']['stage'], stage)
                self.assertEqual(report['error']['reason'], reason)
                self.assertEqual(len(report['workers']), workers)

    def test_user_keys_do_not_collide_with_rng_evidence(self):
        states = []
        original = engine._validate_result
        def capture(result, job, pid):
            observations, count = original(result, job, pid)
            states.extend(item['state'] for item in observations)
            return observations, count
        with patch.object(engine, '_validate_result', side_effect=capture):
            report = audit_resume(FACTORY, factory_config={'colliding_keys': True},
                                  steps=2, split=1)
        self.assertEqual(report['status'], 'PASS', report['error'])
        self.assertEqual(len(states), 10)
        for state in states:
            counter = state['$/s:session/s:counter']
            self.assertEqual(state['$/s:session/s:session/s:counter'], counter)
            self.assertEqual(state['$/s:session/s:global_rng/s:counter'], counter)
            self.assertEqual(state['$/s:global_rng'], {'type': 'dict', 'length': 3})

    def test_saved_global_rng_passes_with_constant_loss(self):
        before = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        report = self.audit('none')
        self.assertEqual(report['status'], 'PASS', report['error'])
        self.assert_constant_loss(report)
        self.assertIsNone(report['first_divergence'])
        self.assertTrue(all(item['matched'] for item in report['comparisons']))
        self.assertEqual(random.getstate(), before[0])
        after = np.random.get_state()
        self.assertEqual(after[0], before[1][0])
        self.assertTrue(np.array_equal(after[1], before[1][1]))
        self.assertEqual(after[2:], before[1][2:])
        self.assertTrue(torch.equal(torch.get_rng_state(), before[2]))


if __name__ == '__main__':
    unittest.main()
