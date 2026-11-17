"""Actual training controls: these fail if persisted training state is omitted."""
import copy
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

from resumetrace.demo import make_session


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def assert_tree_equal(case, expected, actual):
    case.assertIs(type(expected), type(actual))
    if isinstance(expected, torch.Tensor):
        case.assertEqual(expected.dtype, actual.dtype)
        case.assertEqual(expected.shape, actual.shape)
        case.assertTrue(torch.equal(expected, actual))
    elif isinstance(expected, dict):
        case.assertEqual(set(expected), set(actual))
        for key in expected:
            assert_tree_equal(case, expected[key], actual[key])
    elif isinstance(expected, (list, tuple)):
        case.assertEqual(len(expected), len(actual))
        for left, right in zip(expected, actual):
            assert_tree_equal(case, left, right)
    else:
        case.assertEqual(expected, actual)


class DemoTests(unittest.TestCase):
    def setUp(self):
        self.python_rng = random.getstate()
        self.numpy_rng = np.random.get_state()
        self.torch_rng = torch.get_rng_state().clone()
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        random.setstate(self.python_rng)
        np.random.set_state(self.numpy_rng)
        torch.set_rng_state(self.torch_rng)
        torch.set_num_threads(self.threads)

    def continuation(self, fault, split=4, steps=8):
        seed(7)
        session = make_session({'fault': fault})
        for _ in range(split):
            session.step()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session.save(root)
            self.assertTrue(any(path.stat().st_size for path in root.iterdir()))
            saved = copy.deepcopy(session.observe())
            losses = [session.step()['loss'] for _ in range(steps - split)]
            final = copy.deepcopy(session.observe())
            seed(991)
            fresh = make_session({'fault': fault})
            fresh.load(root)
            restored = copy.deepcopy(fresh.observe())
            resumed_losses = [fresh.step()['loss'] for _ in range(steps - split)]
            resumed_final = copy.deepcopy(fresh.observe())
        return saved, restored, losses, resumed_losses, final, resumed_final

    def test_actual_disk_resume_matches_for_multiple_boundaries(self):
        for split in (1, 4, 7):
            with self.subTest(split=split):
                before, after, losses, resumed, final, recovered = self.continuation('none', split)
                assert_tree_equal(self, before, after)
                self.assertEqual(losses, resumed)
                assert_tree_equal(self, final, recovered)

    def test_module_cache_can_mask_missing_checkpoint_optimizer_in_same_process(self):
        # This positive control is essential: cold-process tests must later expose it.
        before, after, losses, resumed, final, recovered = self.continuation('process-cache')
        assert_tree_equal(self, before, after)
        self.assertEqual(losses, resumed)
        assert_tree_equal(self, final, recovered)

    def test_omissions_affect_real_training_not_precomputed_results(self):
        for fault in ('optimizer', 'rng', 'cursor'):
            with self.subTest(fault=fault):
                _, _, expected, actual, _, _ = self.continuation(fault)
                self.assertNotEqual(expected, actual, fault)

    def test_scheduler_omission_changes_loss_at_unaligned_boundary(self):
        _, _, expected, actual, _, _ = self.continuation('scheduler', split=3)
        self.assertNotEqual(expected, actual)

    def test_aligned_scheduler_can_hide_state_loss_from_loss_only_comparison(self):
        before, after, expected, actual, _, _ = self.continuation('scheduler', split=4)
        # Period 2 and split 4 align: optimizer restores LR, so losses alone
        # cannot catch the lost epoch. The audit must compare declared state.
        self.assertEqual(expected, actual)
        self.assertNotEqual(before['scheduler']['last_epoch'], after['scheduler']['last_epoch'])
        normal, restored, _, _, _, _ = self.continuation('none', split=4)
        self.assertEqual(normal['scheduler']['last_epoch'], restored['scheduler']['last_epoch'])

    def test_checkpoint_does_not_alias_later_model_updates(self):
        seed(3)
        session = make_session({'fault': 'none'})
        session.step()
        before = copy.deepcopy(session.observe()['model'])
        with tempfile.TemporaryDirectory() as directory:
            session.save(Path(directory))
            for _ in range(3):
                session.step()
            session.load(Path(directory))
        assert_tree_equal(self, before, session.observe()['model'])

    def test_observation_does_not_consume_rng_or_train(self):
        seed(4)
        session = make_session({})
        before = torch.get_rng_state().clone()
        state = copy.deepcopy(session.observe())
        after = session.observe()
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        assert_tree_equal(self, state, after)
        self.assertEqual(session.step()['step'], 1)

    def test_cursor_uses_real_next_batch_and_survives_epoch_boundary(self):
        before, after, losses, resumed, final, recovered = self.continuation('none', 10, 15)
        self.assertEqual(losses, resumed)
        assert_tree_equal(self, before['data'], after['data'])
        assert_tree_equal(self, final['data'], recovered['data'])

    def test_unknown_fault_and_config_are_rejected(self):
        for config in ({'fault': 'pretend'}, {'other': 1}, {'fault': 1}):
            with self.subTest(config=config), self.assertRaises((TypeError, ValueError)):
                make_session(config)


if __name__ == '__main__':
    unittest.main()
