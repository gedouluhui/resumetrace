"""Re-run the eight release cases from a checkout; never overwrite old evidence.

PYTHONPATH=src python scripts/verify_examples.py --out artifacts/my-verification
This takes multiple fresh Python interpreters. No model downloads or API calls.
"""
import argparse
import hashlib
import json
from pathlib import Path

import resumetrace
from resumetrace import audit_resume
from resumetrace.report import write_reports

ROOT = Path(__file__).resolve().parents[1]


def source_hashes():
    paths = sorted((ROOT / 'src' / 'resumetrace').glob('*.py'))
    return {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if Path(resumetrace.__file__).resolve() != ROOT / 'src' / 'resumetrace' / '__init__.py':
        raise RuntimeError('Use this checkout: set PYTHONPATH=src, not a different installed package')
    args.out.mkdir(parents=True, exist_ok=False)
    before = source_hashes()
    cases = [(fault, fault, 8, 4, 7, 991) for fault in
             ('none', 'optimizer', 'rng', 'cursor', 'scheduler', 'process-cache')]
    cases += [('normal-seed17-split' + str(split), 'none', 5, split, 17, 39) for split in (1, 4)]
    rows = []
    for name, fault, steps, split, seed, restart_seed in cases:
        result = audit_resume('resumetrace.demo:make_session', factory_config={'fault': fault},
                              steps=steps, split=split, seed=seed, restart_seed=restart_seed)
        write_reports(result, json_path=args.out / (name + '.json'), html_path=args.out / (name + '.html'))
        expected = 'PASS' if fault == 'none' else 'DRIFT'
        row = {'case': name, 'status': result['status'], 'control': result['control']['status'],
               'first_divergence': result['first_divergence'], 'error': result['error']}
        rows.append(row)
        print(json.dumps(row, allow_nan=False), flush=True)
        if (result['status'] != expected or result['control']['status'] != 'PASS'
                or result['error'] is not None or len(result['workers']) != 4
                or len(result['comparisons']) != steps - split + 1):
            raise RuntimeError('Acceptance case failed: ' + name)
    if before != source_hashes():
        raise RuntimeError('Source changed during verification; evidence is not accepted')
    record = {'source_sha256': before,
              'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'cases': rows}
    with (args.out / 'acceptance.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print('ALL_RECORDED_CASES_ACCEPTED', len(rows), flush=True)


if __name__ == '__main__':
    main()
