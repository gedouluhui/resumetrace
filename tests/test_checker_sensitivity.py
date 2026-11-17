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


if __name__ == '__main__':
    unittest.main()
