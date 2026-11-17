"""Report boundary tests: escape user text and never overwrite by default."""
import json
import math
import re
from pathlib import Path
import tempfile
import unittest

from resumetrace.report import render_html, write_reports


def sample_report():
    return {
        'schema_version': 1, 'status': 'DRIFT',
        'configuration': {'steps': 8, 'split': 4, 'seed': 7, 'restart_seed': 991},
        'environment': {'python': '3.12', 'pytorch': '2.12'},
        'control': {'status': 'PASS', 'steps_checked': 8},
        'workers': [{'role': role, 'pid': index + 100, 'exit_code': 0, 'token': str(index) * 32}
                    for index, role in enumerate(('reference', 'control', 'prefix', 'resume'))],
        'first_divergence': {'phase': 'restore', 'step': 4, 'paths': ['optimizer/state']},
        'comparisons': [{'phase': 'restore', 'step': 4, 'matched': False,
                         'differing_path_count': 1,
                         'differences': [{'path': 'optimizer/state', 'expected': {'sha256': 'a' * 64},
                                          'actual': {'sha256': 'b' * 64}}],
                         'reference_metrics': {}, 'resumed_metrics': {}},
                        {'phase': 'step', 'step': 5, 'matched': False, 'differing_path_count': 1,
                         'differences': [], 'reference_metrics': {'loss': 0.4},
                         'resumed_metrics': {'loss': 0.7}}],
        'checkpoint': {'file_count': 1, 'total_bytes': 200, 'sha256': 'd' * 64,
                       'files': [{'path_sha256': 'e' * 64, 'bytes': 200, 'sha256': 'c' * 64}]},
        'coverage': {'requested_steps': 8, 'split': 4, 'prefix_steps_checked': 4,
                     'resumed_steps_checked': 4, 'trace_leaves': 300,
                     'scope': 'Exact, observed state only. CPU eager.',
                     'limitations': ['Finite horizon is not future equivalence.']}, 'error': None,
    }


class ReportTests(unittest.TestCase):
    def test_malicious_observation_keys_are_not_executable(self):
        report = sample_report()
        attack = '</script><img src=x onerror=alert(1)>'
        report['comparisons'][0]['differences'][0]['path'] = attack
        report['first_divergence']['paths'] = [attack]
        text = render_html(report)
        self.assertNotIn(attack, text)
        self.assertIn('&lt;/script&gt;', text)
        self.assertNotIn('<script', text.lower())
        self.assertNotIn('src="https://', text)
        self.assertNotIn('href="https://', text)
        self.assertIn('DRIFT', text)
        self.assertIn('Control', text)
        self.assertIn('e' * 64, text)
        self.assertIn('Path SHA-256', text)

    def test_valid_and_invalid_statuses_have_distinct_meanings(self):
        report = sample_report()
        report['status'] = 'INVALID'
        report['error'] = {'stage': 'resume', 'type': 'TimeoutError', 'reason': 'worker_timeout'}
        html = render_html(report)
        self.assertIn('Comparison not valid', html)
        self.assertIn('worker_timeout', html)
        report['status'] = 'PASS'
        report['error'] = None
        report['first_divergence'] = None
        self.assertIn('Observed continuation matched', render_html(report))

    def test_invalid_early_result_with_no_checkpoint_still_has_a_report(self):
        report = sample_report()
        report.update(status='INVALID', checkpoint=None, comparisons=[], first_divergence=None,
                      error={'stage': 'factory', 'type': 'ImportError', 'reason': 'callback_error'})
        report['control'] = {'status': 'NOT_RUN', 'steps_checked': 0}
        output = render_html(report)
        self.assertIn('Comparison not valid', output)
        self.assertIn('No validated checkpoint files', output)
        self.assertNotIn('>None<', output)

    def test_json_and_html_are_actual_saved_files(self):
        report = sample_report()
        with tempfile.TemporaryDirectory() as directory:
            json_path = Path(directory) / 'report.json'
            html_path = Path(directory) / 'report.html'
            write_reports(report, json_path=json_path, html_path=html_path)
            self.assertEqual(json.loads(json_path.read_text(encoding='utf-8')), report)
            self.assertIn('optimizer/state', html_path.read_text(encoding='utf-8'))
            self.assertNotIn(b'\r\n', html_path.read_bytes())

    def test_existing_outputs_are_preserved_until_explicit_force(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'report.json'
            path.write_text('original', encoding='utf-8')
            with self.assertRaises(FileExistsError):
                write_reports(sample_report(), json_path=path)
            self.assertEqual(path.read_text(encoding='utf-8'), 'original')
            write_reports(sample_report(), json_path=path, force=True)
            self.assertEqual(json.loads(path.read_text(encoding='utf-8'))['status'], 'DRIFT')

    def test_equal_destinations_and_nonfinite_data_are_rejected_without_write(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'same'
            with self.assertRaises(ValueError):
                write_reports(sample_report(), json_path=path, html_path=path)
            self.assertFalse(path.exists())
            report = sample_report()
            report['comparisons'][1]['reference_metrics']['loss'] = float('nan')
            with self.assertRaises(ValueError):
                write_reports(report, json_path=path)
            self.assertFalse(path.exists())

    def test_extreme_finite_losses_produce_finite_in_bounds_svg(self):
        for reference, resumed in ((1e15, 1e15), (-1e308, 1e308), (1.79e308, 1.79e308), (0.0, 0.0)):
            with self.subTest(reference=reference, resumed=resumed):
                report = sample_report()
                report['comparisons'][1]['reference_metrics']['loss'] = reference
                report['comparisons'][1]['resumed_metrics']['loss'] = resumed
                output = render_html(report)
                svg = re.search(r'<svg.*?</svg>', output, re.S).group(0)
                for attribute in re.findall(r'(?:points|cx|cy)="([^"]+)"', svg):
                    coordinates = [float(value) for value in re.split(r'[ ,]+', attribute)]
                    self.assertTrue(all(math.isfinite(value) and 0 <= value <= 600 for value in coordinates))
                self.assertNotRegex(svg.lower(), r'(?<![a-z])(nan|inf)(?![a-z])')

    def test_matching_non_loss_metrics_are_inspectable(self):
        report = sample_report()
        report['status'] = 'PASS'
        report['first_divergence'] = None
        for comparison in report['comparisons']:
            comparison.update(matched=True, differing_path_count=0, differences=[],
                              reference_metrics={'validation_accuracy': 0.875},
                              resumed_metrics={'validation_accuracy': 0.875})
        output = render_html(report)
        self.assertIn('validation_accuracy', output)
        self.assertIn('0.875', output)

    def test_unknown_schema_or_status_is_not_presented_as_valid(self):
        for key, value in (('schema_version', 7), ('status', 'unknown')):
            report = sample_report()
            report[key] = value
            with self.assertRaises(ValueError):
                render_html(report)


if __name__ == '__main__':
    unittest.main()
