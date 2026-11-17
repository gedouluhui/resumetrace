"""CLI input/exit-code and saved-output integration tests."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from resumetrace.cli import _prepare, build_parser, main


class CliFastTests(unittest.TestCase):
    def test_parser_preserves_explicit_experiment_settings(self):
        args = build_parser().parse_args(['demo', '--fault', 'cursor', '--steps', '6', '--split', '3'])
        self.assertEqual((args.command, args.fault, args.steps, args.split), ('demo', 'cursor', 6, 3))

    def test_unknown_fault_and_missing_command_exit_two(self):
        for argv in ([], ['demo', '--fault', 'fake']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                build_parser().parse_args(argv)
            self.assertEqual(raised.exception.code, 2)

    def test_invalid_configuration_does_not_reach_worker_import(self):
        for text in ('[]', '{broken', '{"x":NaN}', '{"x":Infinity}', '{"nested":{"x":1e999}}'):
            with self.subTest(text=text), contextlib.redirect_stderr(io.StringIO()):
                args = build_parser().parse_args(['audit', 'does_not_exist:factory', '--config', text])
                with self.assertRaises(ValueError):
                    _prepare(args)
                self.assertEqual(main(['audit', 'does_not_exist:factory', '--config', text]), 2)

    def test_invalid_split_is_a_usage_failure(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(['demo', '--steps', '4', '--split', '4']), 2)


class CliIntegrationTests(unittest.TestCase):
    def test_subprocess_outputs_match_status_and_do_not_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            for fault, expected_code, expected_status in [('none', 0, 'PASS'), ('process-cache', 1, 'DRIFT')]:
                with self.subTest(fault=fault):
                    json_path = Path(directory) / (fault + '.json')
                    html_path = Path(directory) / (fault + '.html')
                    command = [sys.executable, '-m', 'resumetrace', 'demo', '--fault', fault,
                               '--steps', '4', '--split', '2', '--json', str(json_path), '--html', str(html_path)]
                    run = subprocess.run(command, capture_output=True, text=True, timeout=300)
                    self.assertEqual(run.returncode, expected_code, run.stderr + run.stdout)
                    report = json.loads(json_path.read_text(encoding='utf-8'))
                    self.assertEqual(report['status'], expected_status)
                    self.assertIn(expected_status, html_path.read_text(encoding='utf-8'))
                    self.assertNotIn(str(Path(directory)), json_path.read_text(encoding='utf-8'))
                    original = json_path.read_bytes()
                    again = subprocess.run(command, capture_output=True, text=True, timeout=300)
                    self.assertEqual(again.returncode, 2)
                    self.assertEqual(json_path.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
