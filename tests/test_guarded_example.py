import importlib.util
import unittest

import resumetrace
from resumetrace import audit_resume
from resumetrace.torch_observer import TorchSession


class GuardedExampleTests(unittest.TestCase):
    def test_public_wrapper_export(self):
        self.assertIs(getattr(resumetrace, 'TorchSession', None), TorchSession)

    def test_real_sgd_example_has_four_process_positive_control(self):
        self.assertIsNotNone(importlib.util.find_spec('examples.guarded_training'))
        result = audit_resume('examples.guarded_training:make_session', steps=3, split=1)
        self.assertEqual(result['status'], 'PASS', result['error'])
        self.assertEqual(result['coverage']['inspection']['gradient_policy'], 'ignore')
        self.assertEqual(len(result['workers']), 4)


if __name__ == '__main__':
    unittest.main()
