"""Regressions for exact, private and bounded observation evidence."""
import json
import unittest
from unittest.mock import patch

import torch

from resumetrace.fingerprint import compare_fingerprints, fingerprint


class FingerprintTests(unittest.TestCase):
    def test_equal_norm_different_values_are_not_equal(self):
        x, y = torch.tensor([1., -1.]), torch.tensor([-1., 1.])
        self.assertEqual(x.norm().item(), y.norm().item())
        result = compare_fingerprints(fingerprint({'w': x}), fingerprint({'w': y}))
        self.assertFalse(result['matched'])
        self.assertEqual(result['differing_path_count'], 1)
        self.assertEqual(result['differences'][0]['path'], '$/s:w')

    def test_typed_keys_escaping_and_container_evidence(self):
        variants = [{1: 3}, {'1': 3}, {'a/b': 3}, {'a': {'b': 3}},
                    [], (), {}, None, True, 1, 1.0, '1']
        encoded = [json.dumps(fingerprint(x), sort_keys=True) for x in variants]
        self.assertEqual(len(set(encoded)), len(encoded))
        self.assertIn('$/s:a~1b~0c', fingerprint({'a/b~c': 1}))
        self.assertEqual(fingerprint({2: 0, 'a': 1}), fingerprint({'a': 1, 2: 0}))

    def test_scalar_contents_not_in_evidence(self):
        evidence = json.dumps(fingerprint({'secret': 'private-payload-123'}))
        self.assertNotIn('private-payload-123', evidence)
        self.assertIn('sha256', evidence)

    def test_logical_noncontiguous_values_and_no_mutation(self):
        tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
        before = tensor.clone()
        stride = tensor.stride()
        self.assertFalse(tensor.is_contiguous())
        self.assertEqual(fingerprint(tensor), fingerprint(tensor.contiguous()))
        self.assertTrue(torch.equal(tensor, before))
        self.assertEqual(tensor.stride(), stride)
        self.assertNotEqual(fingerprint(tensor), fingerprint(tensor.double()))
        self.assertNotEqual(fingerprint(tensor), fingerprint(tensor.reshape(-1)))
        self.assertEqual(fingerprint(torch.tensor(1.)), fingerprint(torch.tensor(1.)))
        self.assertEqual(fingerprint(torch.empty(0)), fingerprint(torch.empty(0)))
        self.assertEqual(fingerprint(torch.tensor([1.], dtype=torch.bfloat16)),
                         fingerprint(torch.tensor([1.], dtype=torch.bfloat16)))

    def test_cycles_depth_unsupported_and_nonfinite(self):
        cycle = []; cycle.append(cycle)
        deep = 0
        for _ in range(66):
            deep = [deep]
        invalid = [cycle, deep, {False: 1}, {1.1: 0}, set(), object(),
                   float('nan'), float('inf'), torch.tensor([float('inf')]),
                   torch.tensor([complex(0, float('nan'))]),
                   torch.eye(2).to_sparse(), torch.empty(2, device='meta')]
        for state in invalid:
            with self.subTest(kind=type(state).__name__):
                with self.assertRaises((TypeError, ValueError)):
                    fingerprint(state)

    def test_nested_tensor_is_rejected_as_unsupported(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            nested = torch.nested.nested_tensor([torch.ones(2), torch.ones(3)])
        with self.assertRaises((TypeError, ValueError)):
            fingerprint(nested)

    def test_leaf_budget_checked_before_sorting_oversized_container(self):
        with patch('resumetrace.fingerprint._children', side_effect=AssertionError('sorted too early')):
            with self.assertRaises(ValueError):
                fingerprint({'a': 0, 'b': 0, 'c': 0}, max_leaves=2)

    def test_shared_references_are_not_cycles(self):
        shared = {'a': torch.tensor([1])}
        self.assertEqual(fingerprint([shared, shared]),
                         fingerprint([{'a': torch.tensor([1])}, {'a': torch.tensor([1])}]))

    def test_all_tensor_budgets_checked_before_contiguous(self):
        tensors = [torch.ones(4), torch.ones(4)]
        with patch.object(torch.Tensor, 'contiguous', side_effect=AssertionError('allocated')):
            with self.assertRaises(ValueError):
                fingerprint(tensors, max_tensor_bytes=31)
            with self.assertRaises(ValueError):
                fingerprint(tensors, max_leaves=2)
        self.assertEqual(len(fingerprint(tensors, max_tensor_bytes=32, max_leaves=3)), 3)

    def test_budget_types_and_full_difference_count(self):
        for kwargs in ({'max_leaves': True}, {'max_leaves': 0}, {'max_tensor_bytes': -1}):
            with self.assertRaises((TypeError, ValueError)):
                fingerprint(None, **kwargs)
        result = compare_fingerprints(fingerprint(list(range(70))),
                                      fingerprint(list(range(1, 71))))
        self.assertFalse(result['matched'])
        self.assertEqual(result['differing_path_count'], 70)
        self.assertEqual(len(result['differences']), 50)
        self.assertTrue(compare_fingerprints(fingerprint({}), fingerprint({}))['matched'])


if __name__ == '__main__':
    unittest.main()
