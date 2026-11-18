"""Falsify the comparator test in a disposable source copy, never the live tree."""
import hashlib
import importlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


class CheckerSensitivityTests(unittest.TestCase):
    def test_always_equal_mutation_is_caught_by_named_tensor_regression(self):
        module = importlib.import_module('resumetrace.fingerprint')
        original_path = Path(module.__file__)
        original = original_path.read_bytes()
        anchor = "'matched': count == 0"
        text = original.decode('utf-8')
        self.assertEqual(text.count(anchor), 1, 'mutation anchor must be unique')
        mutated = text.replace(anchor, "'matched': True", 1).encode('utf-8')
        self.assertNotEqual(mutated, original, 'NOP mutation is not evidence')
        with tempfile.TemporaryDirectory(prefix='resumetrace-comparator-control-') as directory:
            root = Path(directory)
            package = root / 'resumetrace'
            package.mkdir()
            (package / '__init__.py').write_text('', encoding='utf-8')
            copied_source = package / 'fingerprint.py'
            copied_source.write_bytes(original)
            tests = root / 'tests'
            tests.mkdir()
            shutil.copyfile(Path(__file__).with_name('test_fingerprint.py'), tests / 'test_fingerprint.py')
            command = [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests',
                       '-p', 'test_fingerprint.py', '-k', 'equal_norm', '-v']
            env = {**os.environ, 'PYTHONPATH': str(root), 'PYTHONDONTWRITEBYTECODE': '1'}
            baseline = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(baseline.returncode, 0, baseline.stdout + baseline.stderr)
            copied_source.write_bytes(mutated)
            broken = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, timeout=60)
            evidence = broken.stdout + broken.stderr
            self.assertEqual(broken.returncode, 1, evidence)
            self.assertIn('FAIL: test_equal_norm_different_values_are_not_equal', evidence)
            self.assertIn('True is not false', evidence)
            self.assertNotIn('ERROR:', evidence)
        self.assertEqual(hashlib.sha256(original_path.read_bytes()).digest(), hashlib.sha256(original).digest(),
                         'live source must remain untouched')


class InspectionSensitivityTests(unittest.TestCase):
    def assert_mutation_caught(self, anchor, replacement, test_name):
        import resumetrace.torch_observer as observer
        live = Path(observer.__file__).resolve()
        source = live.read_bytes()
        text = source.decode('utf-8')
        self.assertEqual(text.count(anchor), 1, 'mutation anchor must be unique')
        mutant = text.replace(anchor, replacement, 1).encode('utf-8')
        self.assertNotEqual(mutant, source, 'NOP mutation is not evidence')
        with tempfile.TemporaryDirectory(prefix='resumetrace-inspection-control-') as directory:
            root = Path(directory)
            shutil.copytree(live.parent, root / 'src' / 'resumetrace',
                            ignore=shutil.ignore_patterns('__pycache__'))
            tests = root / 'tests'
            tests.mkdir()
            (tests / '__init__.py').write_text('', encoding='utf-8')
            shutil.copyfile(Path(__file__).with_name('test_torch_observer.py'), tests / 'test_torch_observer.py')
            target = root / 'src' / 'resumetrace' / 'torch_observer.py'
            command = [sys.executable, '-m', 'unittest',
                       'tests.test_torch_observer.TorchObserverTests.' + test_name, '-v']
            env = {**os.environ, 'PYTHONPATH': str(root / 'src'),
                   'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONIOENCODING': 'utf-8'}
            baseline = subprocess.run(command, cwd=root, env=env, capture_output=True,
                                      text=True, encoding='utf-8', timeout=60)
            self.assertEqual(baseline.returncode, 0, baseline.stdout + baseline.stderr)
            target.write_bytes(mutant)
            broken = subprocess.run(command, cwd=root, env=env, capture_output=True,
                                    text=True, encoding='utf-8', timeout=60)
            evidence = broken.stdout + broken.stderr
            self.assertEqual(broken.returncode, 1, evidence)
            self.assertIn('FAIL: ' + test_name, evidence)
            self.assertNotIn('ERROR:', evidence, 'import/runtime crashes are not caught mutations')
        self.assertEqual(live.read_bytes(), source, 'live observer must remain unchanged')

    def test_serialized_slot_identity_mutation_is_caught(self):
        self.assert_mutation_caught('name = names[identity]', "name = 'slot' + str(len(group_names))",
                                    'test_real_parameter_binding_not_serialized_ids')

    def test_missing_observer_guard_mutation_is_caught(self):
        self.assert_mutation_caught("self._check(before, after, 'observer_state_changed')", 'pass',
                                    'test_saturating_observer_changes_weight_before_return')

    def test_missing_save_guard_mutation_is_caught(self):
        self.assert_mutation_caught("self._check(before, after, 'save_state_changed')", 'pass',
                                    'test_valid_checkpoint_with_save_side_effect_has_evidence')

    def test_missing_cross_state_alias_guard_mutation_is_caught(self):
        self.assert_mutation_caught("check_state({'state': state, 'groups': groups, 'gradients': gradient_state})",
                                    'pass', 'test_callback_cannot_alias_optimizer_state_to_parameter')

    def test_missing_internal_overlap_guard_mutation_is_caught(self):
        self.assert_mutation_caught('if stride < span:', 'if False:',
                                    'test_internal_overlap_introduced_by_callback_is_rejected')


if __name__ == '__main__':
    unittest.main()
