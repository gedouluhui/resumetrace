"""Validate new optional evidence without relaxing the existing wire contract."""
import copy
import unittest
from unittest.mock import patch

from resumetrace import audit_resume, engine


class InspectionProtocolTests(unittest.TestCase):
    def setUp(self):
        self.job = {'token': 'a' * 32, 'role': 'prefix', 'start': 0, 'end': 2}
        self.detail = {'check': 'save', 'step': 2, 'paths': ['$/s:torch/s:model/s:parameters/s:x'],
                       'differing_path_count': 1}
        self.result = {'schema_version': 1, 'token': 'a' * 32, 'role': 'prefix', 'pid': 123,
                       'observations': [], 'error': {'stage': 'save', 'type': 'ValueError',
                       'reason': 'save_state_changed', 'detail': self.detail},
                       'inspection': {'mode': 'torch', 'gradient_policy': 'ignore', 'declared_state': True}}

    def test_valid_purity_failure_transports_value_free_evidence(self):
        try:
            engine._validate_result(self.result, self.job, 123)
        except engine._Failure as failure:
            self.assertEqual(failure.error['detail'], self.detail)
            self.assertEqual(failure.error['reason'], 'save_state_changed')
        except Exception as error:
            self.fail('valid evidence rejected: ' + type(error).__name__ + ':' + str(error))
        else:
            self.fail('a purity violation did not reject the experiment')

    def test_malformed_details_and_declarations_rejected(self):
        changes = [lambda r: r['inspection'].update(gradient_policy='guess'),
                   lambda r: r['inspection'].update(declared_state=1),
                   lambda r: r['inspection'].update(extra='secret'),
                   lambda r: r['error']['detail'].update(step=True),
                   lambda r: r['error']['detail'].update(step=1),
                   lambda r: r['error']['detail'].update(check='observe'),
                   lambda r: r['error']['detail'].update(paths=[]),
                   lambda r: r['error']['detail'].update(paths=['C:/private/file']),
                   lambda r: r['error']['detail'].update(paths=['$/' + 'x' * 4097]),
                   lambda r: r['error']['detail'].update(differing_path_count=0),
                   lambda r: r['error']['detail'].update(values=[1, 2]),
                   lambda r: r.pop('inspection'),
                   lambda r: r['error'].pop('detail')]
        for mutate in changes:
            value = copy.deepcopy(self.result)
            mutate(value)
            with self.subTest(mutate=changes.index(mutate)), self.assertRaises(ValueError):
                engine._validate_result(value, self.job, 123)

    def test_changed_policy_between_workers_invalidates_comparison(self):
        original = engine._read_json
        def alter(path, budget):
            value = original(path, budget)
            if value.get('role') == 'control' and 'inspection' in value:
                value['inspection']['gradient_policy'] = 'capture'
            return value
        with patch.object(engine, '_read_json', side_effect=alter):
            result = audit_resume('tests.fixtures.guarded:make_session', steps=2, split=1)
        self.assertEqual(result['status'], 'INVALID')
        self.assertEqual(result['error']['reason'], 'inspection_mode_changed')
        self.assertEqual(len(result['workers']), 2)


if __name__ == '__main__':
    unittest.main()
