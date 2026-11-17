"""Independent struct-packed byte oracles for the supported numeric evidence."""
import hashlib
import struct
import sys
import unittest

import torch
from resumetrace.fingerprint import fingerprint


class BinaryOracleTests(unittest.TestCase):
    def test_tensor_hashes_match_independently_encoded_logical_values(self):
        endian = '<' if sys.byteorder == 'little' else '>'
        cases = [
            ('float32', torch.tensor([1., -0.], dtype=torch.float32), struct.pack(endian + '2f', 1., -0.)),
            ('float64', torch.tensor([1.25, -2.5], dtype=torch.float64), struct.pack(endian + '2d', 1.25, -2.5)),
            ('float16', torch.tensor([1.5, -0.], dtype=torch.float16), struct.pack(endian + '2e', 1.5, -0.)),
            ('bfloat16', torch.tensor([1., -0.], dtype=torch.bfloat16), struct.pack(endian + '2H', 0x3f80, 0x8000)),
            ('complex-conjugate', torch.tensor([complex(1, 2)], dtype=torch.complex64).conj(),
             struct.pack(endian + '2f', 1., -2.)),
            ('uint64', torch.tensor([2**63 + 5], dtype=torch.uint64), struct.pack(endian + 'Q', 2**63 + 5)),
        ]
        for name, tensor, raw in cases:
            with self.subTest(name=name):
                evidence = fingerprint(tensor)['$']
                self.assertEqual(evidence['bytes'], len(raw))
                self.assertEqual(evidence['sha256'], hashlib.sha256(raw).hexdigest())
                changed = bytes([raw[0] ^ 1]) + raw[1:]
                self.assertNotEqual(evidence['sha256'], hashlib.sha256(changed).hexdigest())

    def test_exact_policy_distinguishes_signed_zero(self):
        self.assertNotEqual(fingerprint(torch.tensor([0.])), fingerprint(torch.tensor([-0.])))


if __name__ == '__main__':
    unittest.main()
