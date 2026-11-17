"""Exercise the documented user-integration example, not a mock."""
from pathlib import Path
import tempfile
import unittest

import torch
from examples.custom_training import make_session


class ExampleTests(unittest.TestCase):
    def test_documented_factory_restarts_from_actual_disk_checkpoint(self):
        with torch.random.fork_rng(devices=[]), tempfile.TemporaryDirectory() as directory:
            torch.manual_seed(7)
            trainer = make_session({})
            for _ in range(3):
                trainer.step()
            trainer.save(Path(directory))
            expected = [trainer.step()['loss'] for _ in range(3)]
            torch.manual_seed(99)
            restarted = make_session({})
            restarted.load(Path(directory))
            actual = [restarted.step()['loss'] for _ in range(3)]
            self.assertEqual(expected, actual)
            self.assertEqual(trainer.observe()['cursor'], restarted.observe()['cursor'])
            for name, tensor in trainer.observe()['model'].items():
                self.assertTrue(torch.equal(tensor, restarted.observe()['model'][name]))


if __name__ == '__main__':
    unittest.main()
