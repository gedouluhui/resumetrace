"""Real cold-process integration and strict protocol boundary regressions."""
import copy
import json
import os
from pathlib import Path
import random
import stat
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from resumetrace import audit_resume
from resumetrace import engine


FACTORY = 'tests.fixtures.sessions:make'


class EngineTests(unittest.TestCase):
    def audit(self, fault='none', **kwargs):
        return audit_resume(FACTORY, factory_config={'fault': fault},
                            steps=4, split=2, **kwargs)

    def test_real_four_process_pass_and_complete_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            events = Path(directory) / 'events.jsonl'
            report = audit_resume(FACTORY, factory_config={'lifecycle': str(events)},
                                  steps=4, split=2)
            self.assertEqual(report['status'], 'PASS', report['error'])
            log = [json.loads(line) for line in events.read_text().splitlines()]
        self.assertEqual([x['role'] for x in report['workers']],
                         ['reference', 'control', 'prefix', 'resume'])
        self.assertEqual([x['event'] for x in log],
                         ['start', 'exit', 'start', 'exit', 'start', 'save',
                          'exit', 'start', 'load', 'exit'])
        self.assertEqual(sum(x['steps_taken'] for x in log if x['event'] == 'exit'), 12)
        self.assertEqual(len({x['token'] for x in report['workers']}), 4)
        self.assertTrue(all(x['exit_code'] == 0 for x in report['workers']))
        self.assertEqual(len(report['comparisons']), 3)
        self.assertEqual([(x['phase'], x['step']) for x in report['comparisons']],
                         [('restore', 2), ('step', 3), ('step', 4)])
        self.assertEqual(report['control'], {'status': 'PASS', 'steps_checked': 4})
        self.assertIsNone(report['first_divergence'])
        self.assertEqual(set(report), {'schema_version', 'status', 'configuration',
                         'environment', 'control', 'workers', 'first_divergence',
                         'comparisons', 'checkpoint', 'coverage', 'error'})
        self.assertEqual(report['schema_version'], 1)
        self.assertNotIn(str(events), json.dumps(report))
        self.assertGreater(report['checkpoint']['total_bytes'], 0)

    def test_multiple_seeds_and_splits(self):
        report = audit_resume(FACTORY, steps=3, split=1, seed=82, restart_seed=91)
        self.assertEqual(report['status'], 'PASS', report['error'])

    def test_real_state_drift_has_measured_first_path(self):
        report = self.audit('drift')
        self.assertEqual(report['status'], 'DRIFT', report['error'])
        first = report['first_divergence']
        self.assertEqual((first['phase'], first['step']), ('restore', 2))
        self.assertEqual(first['paths'], ['$/state/s:session/s:value'])
        self.assertEqual(first['paths'], [x['path'] for x in report['comparisons'][0]['differences']])
        self.assertEqual(report['control']['status'], 'PASS')

    def test_metric_only_drift_is_not_silently_ignored(self):
        report = self.audit('metric')
        self.assertEqual(report['status'], 'DRIFT', report['error'])
        self.assertTrue(report['comparisons'][0]['matched'])
        self.assertEqual(report['first_divergence']['step'], 3)
        self.assertIn('$/metrics/s:loss', report['first_divergence']['paths'])

    def test_module_cache_is_lost_across_real_interpreters(self):
        report = self.audit('process-cache')
        self.assertEqual(report['status'], 'DRIFT', report['error'])
        self.assertEqual(report['first_divergence']['paths'], ['$/state/s:session/s:value'])

    def test_control_failure_prevents_save_and_resume(self):
        report = self.audit('control')
        self.assertEqual(report['status'], 'INVALID')
        self.assertEqual(report['error']['reason'], 'nondeterministic_control')
        self.assertEqual(len(report['workers']), 2)
        self.assertIsNone(report['checkpoint'])
        self.assertIsNone(report['first_divergence'])
        self.assertEqual(report['comparisons'], [])

    def test_callback_checkpoint_observer_and_state_failures_are_invalid(self):
        cases = {'throw': 'callback_error', 'save': 'callback_error',
                 'load': 'callback_error', 'empty_checkpoint': 'checkpoint_empty',
                 'zero_checkpoint': 'checkpoint_empty', 'bad_state': 'invalid_state',
                 'cycle': 'invalid_state', 'bad_metrics': 'invalid_metrics',
                 'bool_metrics': 'invalid_metrics',
                 'observer_python': 'observer_rng_consumed',
                 'observer_numpy': 'observer_rng_consumed',
                 'observer_torch': 'observer_rng_consumed', 'crash': 'worker_exit'}
        for fault, reason in cases.items():
            with self.subTest(fault=fault):
                report = self.audit(fault)
                self.assertEqual(report['status'], 'INVALID')
                self.assertEqual(report['error']['reason'], reason)
                self.assertNotIn('PRIVATE_EXCEPTION_TEXT', json.dumps(report))
                self.assertNotIn('C:/private', json.dumps(report))

    def test_budgets_produce_invalid_not_partial_success(self):
        for kwargs in ({'max_tensor_bytes': 5}, {'max_leaves': 1},
                       {'max_trace_leaves': 15}, {'max_checkpoint_bytes': 1}):
            with self.subTest(kwargs=kwargs):
                report = self.audit(**kwargs)
                self.assertEqual(report['status'], 'INVALID')
                self.assertIsNotNone(report['error'])

    def test_timeout_uses_own_child_and_leaves_parent_rng_unchanged(self):
        # 15 seconds allows torch import on CI; fixture sleeps for 90 seconds.
        before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone(),
                  torch.get_num_threads(), torch.are_deterministic_algorithms_enabled(), dict(os.environ))
        report = self.audit('sleep', timeout=15)
        self.assertEqual(report['status'], 'INVALID')
        self.assertEqual(report['error']['reason'], 'worker_timeout')
        self.assertEqual(len(report['workers']), 1)
        self.assertIsNotNone(report['workers'][0]['exit_code'])
        self.assertEqual(random.getstate(), before[0])
        after = np.random.get_state()
        self.assertEqual(after[0], before[1][0])
        self.assertTrue(np.array_equal(after[1], before[1][1]))
        self.assertEqual(after[2:], before[1][2:])
        self.assertTrue(torch.equal(torch.get_rng_state(), before[2]))
        self.assertEqual(torch.get_num_threads(), before[3])
        self.assertEqual(torch.are_deterministic_algorithms_enabled(), before[4])
        self.assertEqual(dict(os.environ), before[5])

    def test_argument_errors_before_spawning(self):
        cases = [{'steps': True}, {'steps': 1}, {'steps': 1001}, {'split': 0},
                 {'split': 8}, {'seed': -1}, {'seed': 2**32}, {'restart_seed': 7},
                 {'timeout': True}, {'timeout': float('inf')}, {'timeout': 0},
                 {'factory_config': []}, {'factory_config': {'x': float('nan')}},
                 {'factory_config': {1: 'x'}}, {'factory_config': {'x': object()}},
                 {'max_leaves': False}, {'max_tensor_bytes': 0}]
        with patch.object(subprocess, 'Popen', side_effect=AssertionError('spawned')):
            for kwargs in cases:
                with self.subTest(kwargs=kwargs):
                    with self.assertRaises((TypeError, ValueError)):
                        audit_resume(FACTORY, **kwargs)
            for factory in ('', 'bad/path:fn', 'module', 'a:b:c', 42, 'a:lambda x: x'):
                with self.subTest(factory=factory):
                    with self.assertRaises((TypeError, ValueError)):
                        audit_resume(factory)

    def test_parent_rejects_corrupted_real_child_output(self):
        original = engine._read_json
        def corrupt(path, limit):
            value = original(path, limit)
            if 'observations' in value:
                value['observations'] = value['observations'][:-1]
            return value
        with patch.object(engine, '_read_json', side_effect=corrupt):
            report = self.audit()
        self.assertEqual(report['status'], 'INVALID')
        self.assertEqual(report['error']['reason'], 'invalid_worker_result')
        self.assertEqual(len(report['workers']), 1)

    def test_invalid_control_output_marks_control_invalid(self):
        original = engine._read_json
        def corrupt(path, limit):
            value = original(path, limit)
            if value.get('role') == 'control':
                value['observations'].pop()
            return value
        with patch.object(engine, '_read_json', side_effect=corrupt):
            report = self.audit()
        self.assertEqual(report['status'], 'INVALID')
        self.assertEqual(report['control']['status'], 'INVALID')
        self.assertEqual(len(report['workers']), 2)

    def test_result_validation_checks_every_identity_and_leaf(self):
        captured = []
        original = engine._validate_result
        def capture(result, job, pid):
            captured.append((copy.deepcopy(result), copy.deepcopy(job), pid))
            return original(result, job, pid)
        with patch.object(engine, '_validate_result', side_effect=capture):
            self.assertEqual(self.audit()['status'], 'PASS')
        result, job, pid = captured[0]
        mutations = [lambda x: x.update(schema_version=True),
                     lambda x: x.update(token='0' * 32),
                     lambda x: x.update(role='resume'),
                     lambda x: x.update(pid=-1),
                     lambda x: x.update(extra=0),
                     lambda x: x['observations'].pop(),
                     lambda x: x['observations'][0].update(step=True),
                     lambda x: x['observations'][0].update(metrics={'loss': 0}),
                     lambda x: x['observations'][1].update(metrics={}),
                     lambda x: x['observations'][1].update(metrics={'loss': False}),
                     lambda x: x['observations'][0]['state']['$/s:session/s:value'].update(sha256='bad'),
                     lambda x: x['observations'][0]['state']['$/s:session/s:rng'].update(bytes=1),
                     lambda x: x['observations'][0]['state']['$'].update(length=999),
                     lambda x: x['observations'][0]['state'].update({'$/s:absent/s:x': {'type': 'none', 'sha256': '0' * 64}}),
                     lambda x: x.update(error={'stage': 'observe', 'type': 'ValueError', 'reason': 'PRIVATE'})]
        for mutate in mutations:
            corrupted = copy.deepcopy(result)
            mutate(corrupted)
            with self.subTest(mutation=mutations.index(mutate)):
                with self.assertRaises((TypeError, ValueError)):
                    original(corrupted, job, pid)

    def test_checkpoint_hash_covers_content_paths_and_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            file = root / 'one.bin'
            file.write_bytes(b'abc')
            first = engine._checkpoint(root, 3)
            file.write_bytes(b'acb')
            second = engine._checkpoint(root, 3)
            self.assertNotEqual(first['sha256'], second['sha256'])
            file.rename(root / 'two.bin')
            third = engine._checkpoint(root, 3)
            self.assertNotEqual(second['sha256'], third['sha256'])
            self.assertNotIn(str(root), json.dumps(third))
            with self.assertRaises(engine._Failure) as caught:
                engine._checkpoint(root, 2)
            self.assertEqual(caught.exception.error['reason'], 'checkpoint_byte_limit')
            for index in range(1000):
                (root / str(index)).write_bytes(b'x')
            with self.assertRaises(engine._Failure) as caught:
                engine._checkpoint(root, 10000)
            self.assertEqual(caught.exception.error['reason'], 'checkpoint_file_limit')

    def test_checkpoint_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'state').write_bytes(b'x')
            try:
                (root / 'link').symlink_to(root / 'state')
            except OSError:
                self.skipTest('OS does not permit test symlink creation')
            with self.assertRaises(engine._Failure) as caught:
                engine._checkpoint(root, 100)
            self.assertEqual(caught.exception.error['reason'], 'checkpoint_unsafe')

    def test_cleanup_failure_overrides_success(self):
        original = engine.shutil.rmtree
        def cleanup_then_error(path):
            original(path)
            raise OSError('PRIVATE_EXCEPTION_TEXT')
        with patch.object(engine.shutil, 'rmtree', side_effect=cleanup_then_error):
            report = self.audit()
        self.assertEqual(report['status'], 'INVALID')
        self.assertEqual(report['error']['reason'], 'cleanup_failed')
        self.assertNotIn('PRIVATE_EXCEPTION_TEXT', json.dumps(report))

    def test_reader_refuses_nonregular_result_before_opening(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'result.json'
            path.write_text('{}', encoding='utf-8')
            # Windows cannot create a FIFO; simulate only its filesystem metadata.
            metadata = os.stat_result((stat.S_IFIFO, 0, 0, 1, 0, 0, 2, 0, 0, 0))
            with patch.object(Path, 'lstat', return_value=metadata):
                with self.assertRaises(ValueError):
                    engine._read_json(path, 100)

    def test_bounded_json_reader_rejects_duplicate_nonfinite_and_truncated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'result.json'
            for raw in (b'{', b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1e999}', b'x' * 101):
                path.write_bytes(raw)
                with self.subTest(raw=raw[:20]):
                    with self.assertRaises((TypeError, ValueError, OSError)):
                        engine._read_json(path, 100)


if __name__ == '__main__':
    unittest.main()
