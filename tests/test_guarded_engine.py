import tempfile
import unittest

from resumetrace import audit_resume

FACTORY = 'tests.fixtures.guarded:make_session'


class GuardedEngineTests(unittest.TestCase):
    def audit(self, fault='none', **config):
        return audit_resume(FACTORY, factory_config={'fault': fault, **config}, steps=3, split=1)

    def test_healthy_training_keeps_gradients_and_validates_coverage(self):
        report = self.audit()
        self.assertEqual(report['status'], 'PASS', report['error'])
        self.assertEqual(report['coverage'].get('inspection'),
                         {'mode': 'torch', 'gradient_policy': 'ignore', 'declared_state': True})
        self.assertEqual(len(report['workers']), 4)
        self.assertTrue(all(item['matched'] for item in report['comparisons']))

    def test_named_optimizer_binding_drifts_at_restore(self):
        report = self.audit('optimizer-order')
        self.assertEqual(report['status'], 'DRIFT', report['error'])
        first = report['first_divergence']
        self.assertEqual((first['phase'], first['step']), ('restore', 1))
        self.assertTrue(any('/s:torch/s:optimizer/' in p for p in first['paths']), first)

    def test_observer_mutation_is_invalid_with_named_evidence(self):
        report = self.audit('observer')
        self.assertEqual(report['status'], 'INVALID')
        self.assertEqual(report['error']['reason'], 'observer_state_changed')
        detail = report['error']['detail']
        self.assertEqual((detail['check'], detail['step']), ('observe', 0))
        self.assertTrue(any('/s:parameters/' in p for p in detail['paths']))
        self.assertEqual(len(report['workers']), 1)

    def test_save_mutation_has_evidence_without_blame_on_restore(self):
        report = self.audit('save')
        self.assertEqual(report['status'], 'INVALID')
        self.assertEqual(report['error']['reason'], 'save_state_changed')
        self.assertEqual(report['control']['status'], 'PASS')
        detail = report['error']['detail']
        self.assertEqual((detail['check'], detail['step']), ('save', 1))
        self.assertTrue(detail['paths'])
        self.assertEqual(len(report['workers']), 3)
        self.assertIsNone(report['first_divergence'])
        self.assertIsNone(report['checkpoint'])

    def test_generic_valid_checkpoint_not_subject_to_strict_save_purity(self):
        report = self.audit('save', generic=True)
        self.assertEqual(report['status'], 'PASS', report['error'])
        self.assertNotIn('inspection', report['coverage'])

    def test_explicit_gradient_policy_is_not_silently_downgraded(self):
        capture = self.audit(gradient_policy='capture')
        self.assertEqual(capture['status'], 'DRIFT', capture['error'])
        self.assertTrue(any('/s:gradients/' in path for path in capture['first_divergence']['paths']))
        strict = self.audit(gradient_policy='require_none')
        self.assertEqual(strict['status'], 'INVALID')
        self.assertEqual(strict['error']['reason'], 'torch_inspection_unsupported')

    def test_factory_failure_after_reference_keeps_original_reason(self):
        with tempfile.TemporaryDirectory(prefix='resumetrace-late-factory-') as directory:
            report = self.audit('factory-late', marker=str(directory + '/marker'))
        self.assertEqual(report['status'], 'INVALID')
        self.assertEqual(report['error']['reason'], 'callback_error')
        self.assertEqual(report['error']['type'], 'RuntimeError')
        self.assertEqual(len(report['workers']), 2)

    def test_successful_worker_without_matching_declaration_is_invalid(self):
        with tempfile.TemporaryDirectory(prefix='resumetrace-late-wrapper-') as directory:
            report = self.audit('generic-late', marker=str(directory + '/marker'))
        self.assertEqual(report['status'], 'INVALID')
        self.assertEqual(report['error']['reason'], 'inspection_mode_changed')
        self.assertEqual(report['coverage']['inspection']['mode'], 'torch')


if __name__ == '__main__':
    unittest.main()
