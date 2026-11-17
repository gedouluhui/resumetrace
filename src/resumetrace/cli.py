"""Command line for trusted importable training jobs and the included demo."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='resumetrace', description='Test checkpoint continuation across real Python process exits.')
    commands = parser.add_subparsers(dest='command', required=True)
    demo = commands.add_parser('demo', help='Run the included small CPU training job')
    demo.add_argument('--fault', default='none', choices=('none', 'optimizer', 'rng', 'cursor', 'scheduler', 'process-cache'))
    audit = commands.add_parser('audit', help='Run your trusted importable module:function training factory')
    audit.add_argument('factory', help='Importable module:function accepting one config dict')
    audit.add_argument('--config', default='{}', help='JSON object passed to the factory (not a filename)')
    for command in (demo, audit):
        command.add_argument('--steps', type=int, default=8, help='Total optimizer steps (2..1000)')
        command.add_argument('--split', type=int, default=4, help='Checkpoint after this step')
        command.add_argument('--seed', type=int, default=7)
        command.add_argument('--restart-seed', type=int, default=991)
        command.add_argument('--timeout', type=float, default=120, help='Seconds allowed for each Python worker')
        command.add_argument('--json', dest='json_path', type=Path, help='Write machine-readable evidence')
        command.add_argument('--html', dest='html_path', type=Path, help='Write a standalone offline HTML report')
        command.add_argument('--force', action='store_true', help='Explicitly replace existing report files')
    return parser


def _invalid_constant(value):
    raise ValueError('configuration must contain only finite JSON values')


def _prepare(args):
    if not 2 <= args.steps <= 1000 or not 0 < args.split < args.steps:
        raise ValueError('require 2 <= steps <= 1000 and 0 < split < steps')
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError('timeout must be positive and finite')
    if not all(0 <= value < 2**32 for value in (args.seed, args.restart_seed)) or args.seed == args.restart_seed:
        raise ValueError('seeds must be distinct integers in 0..2**32-1')
    if args.command == 'demo':
        factory, config = 'resumetrace.demo:make_session', {'fault': args.fault}
    else:
        if len(args.config.encode('utf-8')) > 65536:
            raise ValueError('configuration exceeds 64 KiB')
        try:
            config = json.loads(args.config, parse_constant=_invalid_constant)
            # A valid exponent such as 1e999 parses to inf without parse_constant.
            json.dumps(config, allow_nan=False)
        except (ValueError, RecursionError) as error:
            raise ValueError('config must be a finite JSON object') from error
        if type(config) is not dict:
            raise ValueError('config must be a JSON object')
        factory = args.factory
    paths = [path.absolute() for path in (args.json_path, args.html_path) if path is not None]
    if len(paths) == 2 and (paths[0].resolve() == paths[1].resolve() or
                            (paths[0].exists() and paths[1].exists() and paths[0].samefile(paths[1]))):
        raise ValueError('JSON and HTML destinations must differ')
    for path in paths:
        if path.is_symlink():
            raise ValueError('report destination must not be a symlink')
        if path.exists() and not args.force:
            raise FileExistsError('report exists; use --force to replace it')
        if not path.parent.is_dir():
            raise FileNotFoundError('report parent directory does not exist')
    return factory, config


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        factory, config = _prepare(args)
        from .engine import audit_resume
        from .report import write_reports
        result = audit_resume(factory, factory_config=config, steps=args.steps, split=args.split,
                              seed=args.seed, restart_seed=args.restart_seed, timeout=args.timeout)
        write_reports(result, json_path=args.json_path, html_path=args.html_path, force=args.force)
    except (TypeError, ValueError, OSError) as error:
        # These are local CLI/config/output failures, not callback exception text.
        print(f'ResumeTrace: {error}', file=sys.stderr)
        return 2
    status = result['status']
    first = result.get('first_divergence')
    suffix = f' | first divergence: {first["phase"]} at step {first["step"]}' if first else ''
    print(f'{status}{suffix}')
    if result.get('error'):
        error = result['error']
        print(f'Comparison invalid: {error.get("stage", "unknown")} / {error.get("reason", "execution_error")}', file=sys.stderr)
    return {'PASS': 0, 'DRIFT': 1, 'INVALID': 2}[status]
