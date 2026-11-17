"""Private bounded-JSON protocol. Each invocation is one training lifecycle."""
import importlib
import os
from pathlib import Path
import random
import re
import sys

import numpy as np
import torch

from .engine import (JOB_BYTES, RESULT_BYTES, ROLES, _Failure, _encode, _metrics,
                     _read_json, _safe_type, _validate_config)
from .fingerprint import fingerprint


def _job(path):
    job = _read_json(path, JOB_BYTES)
    fields = {'schema_version', 'token', 'role', 'configuration', 'start', 'end',
              'checkpoint_dir', 'result_path', 'trace_budget'}
    if type(job) is not dict or set(job) != fields:
        raise ValueError('invalid_job')
    if type(job['schema_version']) is not int or job['schema_version'] != 1:
        raise ValueError('invalid_job_version')
    if job['role'] not in ROLES or type(job['token']) is not str or re.fullmatch('[0-9a-f]{32}', job['token']) is None:
        raise ValueError('invalid_job_identity')
    config = job['configuration']
    if type(config) is not dict or set(config) != {'factory', 'factory_config', 'steps', 'split',
            'seed', 'restart_seed', 'timeout', 'max_tensor_bytes', 'max_leaves',
            'max_trace_leaves', 'max_checkpoint_bytes'}:
        raise ValueError('invalid_job_configuration')
    _validate_config(config['factory'], config)
    start = config['split'] if job['role'] == 'resume' else 0
    end = config['split'] if job['role'] == 'prefix' else config['steps']
    if type(job['start']) is not int or type(job['end']) is not int or (job['start'], job['end']) != (start, end):
        raise ValueError('invalid_job_horizon')
    if type(job['trace_budget']) is not int or not 0 <= job['trace_budget'] <= config['max_trace_leaves']:
        raise ValueError('invalid_job_budget')
    parent = Path(path).absolute().parent
    for name, expected in (('result_path', parent / (job['role'] + '.result.json')),
                           ('checkpoint_dir', parent / 'checkpoint')):
        if type(job[name]) is not str or not Path(job[name]).is_absolute() or Path(job[name]) != expected:
            raise ValueError('invalid_job_path')
    return job


def _rng_state():
    return random.getstate(), np.random.get_state(), torch.get_rng_state()


def _same_rng(before, after):
    return (before[0] == after[0] and before[1][0] == after[1][0]
            and np.array_equal(before[1][1], after[1][1])
            and before[1][2:] == after[1][2:]
            and torch.equal(before[2], after[2]))


def _run(job):
    config, role = job['configuration'], job['role']
    result = {'schema_version': 1, 'token': job['token'], 'role': role,
              'pid': os.getpid(), 'observations': [], 'error': None}
    stage = 'setup'
    try:
        seed = config['restart_seed'] if role == 'resume' else config['seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)
        stage = 'factory'
        module, function = config['factory'].split(':')
        factory = getattr(importlib.import_module(module), function)
        session = factory(config['factory_config'])
        if any(not callable(getattr(session, name, None)) for name in ('step', 'observe', 'save', 'load')):
            raise TypeError('session_protocol')
        if role == 'resume':
            stage = 'load'
            session.load(Path(job['checkpoint_dir']))
        retained, serialized = 0, len(_encode(result))
        for index in range(job['start'], job['end'] + 1):
            metrics = None
            if index != job['start']:
                stage = 'step'
                metrics = session.step()
                try:
                    _metrics(metrics)
                    # Detach the returned mapping from later callback mutations.
                    metrics = dict(metrics)
                except (TypeError, ValueError) as error:
                    raise _Failure('step', 'invalid_metrics', _safe_type(error)) from None
            stage = 'observe'
            before = _rng_state()
            state = session.observe()
            if not _same_rng(before, _rng_state()):
                raise _Failure('observe', 'observer_rng_consumed')
            stage = 'fingerprint'
            try:
                if type(state) is not dict:
                    raise TypeError('observation_type')
                # Reuse the pre-observe snapshot whose purity was just checked.
                # Keep user keys isolated and charge all evidence to one budget.
                numpy_state = before[1]
                evidence = {'session': state, 'global_rng': {
                    'python': before[0],
                    'numpy': (numpy_state[0], torch.from_numpy(numpy_state[1]),
                              *numpy_state[2:]),
                    'torch': before[2]}}
                state = fingerprint(evidence, max_tensor_bytes=config['max_tensor_bytes'],
                                    max_leaves=config['max_leaves'])
            except (TypeError, ValueError, RuntimeError) as error:
                raise _Failure('fingerprint', 'invalid_state', _safe_type(error)) from None
            retained += len(state) + len(metrics or {})
            if retained > job['trace_budget']:
                raise _Failure('result', 'trace_leaf_limit')
            observation = {'step': index, 'state': state, 'metrics': metrics}
            stage = 'result'
            serialized += len(_encode(observation)) + 1
            if serialized > RESULT_BYTES:
                raise _Failure('result', 'result_byte_limit')
            result['observations'].append(observation)
        if role == 'prefix':
            stage = 'save'
            session.save(Path(job['checkpoint_dir']))
    except _Failure as failure:
        result['observations'] = []
        result['error'] = failure.error
    except BaseException as error:
        result['observations'] = []
        reason = 'worker_setup_error' if stage == 'setup' else 'callback_error'
        result['error'] = {'stage': stage, 'type': _safe_type(error), 'reason': reason}
    return result


def main():
    if len(sys.argv) != 2:
        return 2
    try:
        job = _job(Path(sys.argv[1]).absolute())
        result = _run(job)
        payload = _encode(result)
        if len(payload) > RESULT_BYTES:
            result['observations'] = []
            result['error'] = {'stage': 'result', 'type': 'ValueError', 'reason': 'result_byte_limit'}
            payload = _encode(result)
        Path(job['result_path']).write_bytes(payload)
    except BaseException:
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
