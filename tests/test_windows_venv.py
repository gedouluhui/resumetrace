"""A real Windows venv redirector must not break strict worker PID validation."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import venv


@unittest.skipUnless(sys.platform == 'win32', 'Windows venv redirector regression')
class WindowsVenvTests(unittest.TestCase):
    def test_real_venv_audit_keeps_environment_and_owned_worker_identity(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix='resumetrace-venv-control-') as directory:
            environment = Path(directory) / 'environment'
            venv.EnvBuilder(with_pip=False, system_site_packages=True).create(environment)
            python = environment / 'Scripts' / 'python.exe'
            code = (
                "import json,sys; from resumetrace import audit_resume; "
                "r=audit_resume('tests.fixtures.rng_only:make',steps=2,split=1); "
                "print(json.dumps({'status':r['status'],'error':r['error'],"
                "'control':r['control'],'workers':r['workers'],"
                "'is_venv':sys.prefix!=sys.base_prefix}))"
            )
            env = {**os.environ, 'PYTHONPATH': os.pathsep.join((str(root / 'src'), str(root))),
                   'PYTHONDONTWRITEBYTECODE': '1'}
            run = subprocess.run([str(python), '-c', code], cwd=root, env=env,
                                 text=True, capture_output=True, timeout=180)
            self.assertEqual(run.returncode, 0, run.stderr)
            observed = json.loads(run.stdout)
            self.assertTrue(observed['is_venv'])
            self.assertEqual(observed['status'], 'PASS', observed)
            self.assertIsNone(observed['error'])
            self.assertEqual(observed['control'], {'status': 'PASS', 'steps_checked': 2})
            self.assertEqual([item['role'] for item in observed['workers']],
                             ['reference', 'control', 'prefix', 'resume'])


if __name__ == '__main__':
    unittest.main()
