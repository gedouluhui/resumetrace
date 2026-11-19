"""Sequential cold-interpreter audit. No training objects cross this boundary."""
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import uuid

from .fingerprint import DTYPE_SIZES, MAX_PATH, compare_fingerprints, escape, fingerprint
from .rng_format import check as check_rng_format
from .rng_format import templates as rng_templates
from .torch_observer import REASONS as INSPECTION_REASONS, POLICIES


JOB_BYTES = 1048576
RESULT_BYTES = 67108864
ROLES = ('reference', 'control', 'prefix', 'resume')
REASONS = {'callback_error', 'invalid_state', 'invalid_metrics', 'observer_rng_consumed',
           'trace_leaf_limit', 'result_byte_limit', 'worker_setup_error'} | INSPECTION_REASONS
STAGES = {'setup', 'factory', 'step', 'observe', 'fingerprint', 'save', 'load', 'result'}
TYPES = {'TypeError', 'ValueError', 'RuntimeError', 'OSError', 'MemoryError',
         'ImportError', 'AttributeError', 'TimeoutError', 'SystemExit', 'Exception'}


class _Failure(Exception):
    def __init__(self, stage, reason, kind='ValueError', detail=None):
        self.error = {'stage': stage, 'type': kind, 'reason': reason}
        if detail is not None:
            self.error['detail'] = detail


def _safe_type(error):
    name = type(error).__name__
    return name if name in TYPES else 'Exception'


def _json_config(value, depth=0, active=None):
    if depth > 64:
        raise ValueError('config_depth_limit')
    active = set() if active is None else active
    if type(value) in (dict, list):
        if id(value) in active:
            raise ValueError('config_cycle')
        active.add(id(value))
        if type(value) is dict and any(type(k) is not str for k in value):
            raise TypeError('config_key_type')
        for child in (value.values() if type(value) is dict else value):
            _json_config(child, depth + 1, active)
        active.remove(id(value))
    elif type(value) not in (type(None), str, bool, int, float):
        raise TypeError('config_value_type')
    elif type(value) is float and not math.isfinite(value):
        raise ValueError('config_nonfinite')


def _encode(value):
    return json.dumps(value, allow_nan=False, ensure_ascii=True,
                      sort_keys=True, separators=(',', ':')).encode('utf-8')


def _read_json(path, limit):
    """Bound before parsing and reject duplicates and nonstandard JSON numbers."""
    path = Path(path)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 1024:
        raise ValueError('json_file_type')
    if info.st_size > limit:
        raise ValueError('json_byte_limit')
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError('json_byte_limit')
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('json_duplicate_key')
            result[key] = value
        return result
    def number(text):
        value = float(text)
        if not math.isfinite(value):
            raise ValueError('json_nonfinite')
        return value
    def constant(_):
        raise ValueError('json_nonfinite')
    return json.loads(raw, object_pairs_hook=pairs, parse_float=number, parse_constant=constant)


def _validate_config(factory, config):
    if type(factory) is not str:
        raise TypeError('factory_type')
    if len(factory) > 512 or re.fullmatch(r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*', factory, re.ASCII) is None:
        raise ValueError('factory_format')
    for name in ('steps', 'split', 'seed', 'restart_seed', 'max_tensor_bytes',
                 'max_leaves', 'max_trace_leaves', 'max_checkpoint_bytes'):
        if type(config[name]) is not int:
            raise TypeError(name + '_type')
    if not 2 <= config['steps'] <= 1000 or not 1 <= config['split'] < config['steps']:
        raise ValueError('step_range')
    if any(not 0 <= config[name] < 2**32 for name in ('seed', 'restart_seed')) or config['seed'] == config['restart_seed']:
        raise ValueError('seed_range')
    if any(config[name] <= 0 for name in config if name.startswith('max_')):
        raise ValueError('budget_range')
    timeout = config['timeout']
    if type(timeout) not in (int, float):
        raise TypeError('timeout_type')
    try:
        finite_timeout = math.isfinite(timeout)
    except OverflowError:
        finite_timeout = False
    if not finite_timeout or timeout <= 0:
        raise ValueError('timeout_range')
    if type(config['factory_config']) is not dict:
        raise TypeError('config_type')
    _json_config(config['factory_config'])
    if len(_encode(config)) > JOB_BYTES - 8192:
        raise ValueError('config_byte_limit')


def _metrics(value):
    if type(value) is not dict or not value:
        raise ValueError('invalid_metrics')
    for key, item in value.items():
        if type(key) is not str or len(key) > MAX_PATH - 32 or type(item) not in (int, float):
            raise TypeError('invalid_metrics')
        if type(item) is float and not math.isfinite(item):
            raise ValueError('invalid_metrics')
    return value


def _validate_state(state, config):
    if type(state) is not dict or not 1 <= len(state) <= config['max_leaves'] or '$' not in state:
        raise ValueError('invalid_state')
    total = 0
    children = {}
    for path, leaf in state.items():
        if type(path) is not str or len(path) > MAX_PATH or not path.startswith('$') or type(leaf) is not dict:
            raise ValueError('invalid_state')
        if path != '$':
            parent, separator, segment = path.rpartition('/')
            if not separator or parent not in state or path.count('/') > 64:
                raise ValueError('invalid_path')
            parent_kind = state[parent].get('type') if type(state[parent]) is dict else None
            tag, colon, key = segment.partition(':')
            expected_tags = {'dict': ('s', 'i'), 'list': ('l',), 'tuple': ('t',)}
            if not colon or tag not in expected_tags.get(parent_kind, ()):
                raise ValueError('invalid_path')
            if tag == 's':
                if re.search(r'~(?![01])', key):
                    raise ValueError('invalid_path')
            elif re.fullmatch(r'0|-?[1-9][0-9]*', key) is None or (tag in ('l', 't') and int(key) < 0):
                raise ValueError('invalid_path')
            children.setdefault(parent, []).append(key)
        kind = leaf.get('type')
        if kind in ('dict', 'list', 'tuple'):
            if set(leaf) != {'type', 'length'} or type(leaf['length']) is not int or not 0 <= leaf['length'] <= config['max_leaves']:
                raise ValueError('invalid_container')
        else:
            digest = leaf.get('sha256')
            if type(digest) is not str or re.fullmatch('[0-9a-f]{64}', digest) is None:
                raise ValueError('invalid_digest')
            if kind == 'tensor':
                if set(leaf) != {'type', 'dtype', 'shape', 'bytes', 'sha256'} or leaf['dtype'] not in DTYPE_SIZES:
                    raise ValueError('invalid_tensor')
                shape = leaf['shape']
                if type(shape) is not list or len(shape) > 64 or any(type(n) is not int or n < 0 for n in shape):
                    raise ValueError('invalid_shape')
                size = math.prod(shape) * DTYPE_SIZES[leaf['dtype']]
                if type(leaf['bytes']) is not int or leaf['bytes'] != size:
                    raise ValueError('invalid_tensor_bytes')
                total += size
            elif kind not in ('none', 'bool', 'int', 'float', 'str') or set(leaf) != {'type', 'sha256'}:
                raise ValueError('invalid_scalar')
    if total > config['max_tensor_bytes'] or state['$']['type'] != 'dict':
        raise ValueError('invalid_state')
    for path, leaf in state.items():
        if leaf['type'] in ('dict', 'list', 'tuple'):
            keys = children.get(path, [])
            if len(keys) != leaf['length']:
                raise ValueError('invalid_container')
            if leaf['type'] != 'dict' and {int(key) for key in keys} != set(range(leaf['length'])):
                raise ValueError('invalid_sequence')
    # A valid generic fingerprint is not sufficient: every observation must
    # include the session and all three default RNGs, including at restore.
    containers = {'$': ('dict', 2), '$/s:global_rng': ('dict', 3),
                  '$/s:global_rng/s:python': ('tuple', 3),
                  '$/s:global_rng/s:numpy': ('tuple', 5)}
    if any(state.get(path) != {'type': kind, 'length': length}
           for path, (kind, length) in containers.items()):
        raise ValueError('invalid_observation_evidence')
    required_types = {'$/s:session': 'dict', '$/s:global_rng/s:torch': 'tensor',
                      '$/s:global_rng/s:numpy/t:1': 'tensor'}
    if any(state.get(path, {}).get('type') != kind
           for path, kind in required_types.items()):
        raise ValueError('invalid_observation_evidence')
    # Container shape alone cannot prove that a recorded RNG is the state this
    # runtime actually produces: a float32 empty Torch state or a truncated
    # Python state tuple is self-consistent but incomparable. Check the
    # internal structure, dtype and shape against the declared format.
    check_rng_format(state, rng_templates())


def _validate_inspection(value):
    if (type(value) is not dict or set(value) != {'mode', 'gradient_policy', 'declared_state'} or
            value['mode'] != 'torch' or type(value['gradient_policy']) is not str or
            value['gradient_policy'] not in POLICIES or type(value['declared_state']) is not bool):
        raise ValueError('invalid_inspection')


def _validate_inspection_detail(error, job):
    detail = error.get('detail')
    check = 'observe' if error['reason'] == 'observer_state_changed' else 'save'
    if (type(detail) is not dict or set(detail) != {'check', 'step', 'paths', 'differing_path_count'} or
            detail['check'] != check or error['stage'] != check or
            type(detail['step']) is not int or not job['start'] <= detail['step'] <= job['end'] or
            (check == 'save' and (job['role'] != 'prefix' or detail['step'] != job['end']))):
        raise ValueError('invalid_inspection_detail')
    paths, count = detail['paths'], detail['differing_path_count']
    if (type(paths) is not list or not 1 <= len(paths) <= 50 or type(count) is not int or
            count < len(paths) or count > 2000000 or
            any(type(path) is not str or not path.startswith('$/') or len(path) > MAX_PATH
                for path in paths) or len(set(paths)) != len(paths)):
        raise ValueError('invalid_inspection_detail')
    return detail


def _validate_result(result, job, pid):
    required = {'schema_version', 'token', 'role', 'pid', 'observations', 'error'}
    if type(result) is not dict or set(result) not in (required, required | {'inspection'}):
        raise ValueError('invalid_envelope')
    if 'inspection' in result:
        _validate_inspection(result['inspection'])
    if type(result['schema_version']) is not int or result['schema_version'] != 1 or result['token'] != job['token'] or result['role'] != job['role']:
        raise ValueError('invalid_identity')
    if type(result['pid']) is not int or result['pid'] != pid:
        raise ValueError('invalid_identity')
    error, observations = result['error'], result['observations']
    if error is not None:
        basic = {'stage', 'type', 'reason'}
        if (type(error) is not dict or set(error) not in (basic, basic | {'detail'}) or
                any(type(error[key]) is not str for key in basic) or error['stage'] not in STAGES or
                error['type'] not in TYPES or error['reason'] not in REASONS or observations != []):
            raise ValueError('invalid_error')
        if error['reason'] in INSPECTION_REASONS and 'inspection' not in result:
            raise ValueError('invalid_inspection_error')
        if error['reason'] in ('observer_state_changed', 'save_state_changed'):
            detail = _validate_inspection_detail(error, job)
        elif 'detail' in error:
            raise ValueError('invalid_error_detail')
        else:
            detail = None
        raise _Failure(error['stage'], error['reason'], error['type'], detail)
    indices = range(job['start'], job['end'] + 1)
    if type(observations) is not list or len(observations) != len(indices):
        raise ValueError('invalid_horizon')
    count = 0
    for index, observation in zip(indices, observations):
        if type(observation) is not dict or set(observation) != {'step', 'state', 'metrics'} or type(observation['step']) is not int or observation['step'] != index:
            raise ValueError('invalid_observation')
        _validate_state(observation['state'], job['configuration'])
        if index == job['start']:
            if observation['metrics'] is not None:
                raise ValueError('invalid_boundary_metrics')
        else:
            _metrics(observation['metrics'])
        count += len(observation['state']) + len(observation['metrics'] or {})
        if count > job['trace_budget']:
            raise ValueError('invalid_trace_budget')
    return observations, count


def _checkpoint(directory, budget):
    entries, total, seen, nonempty = [], 0, 0, False
    def walk(folder):
        nonlocal total, seen, nonempty
        info = folder.lstat()
        if not stat.S_ISDIR(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 1024:
            raise _Failure('checkpoint', 'checkpoint_unsafe')
        with os.scandir(folder) as iterator:
            for entry in iterator:
                seen += 1
                if seen > 2000:
                    raise _Failure('checkpoint', 'checkpoint_entry_limit')
                info = entry.stat(follow_symlinks=False)
                if entry.is_symlink() or getattr(info, 'st_file_attributes', 0) & 1024:
                    raise _Failure('checkpoint', 'checkpoint_unsafe')
                path = Path(entry.path)
                if stat.S_ISDIR(info.st_mode):
                    walk(path)
                elif stat.S_ISREG(info.st_mode):
                    if len(entries) >= 1000:
                        raise _Failure('checkpoint', 'checkpoint_file_limit')
                    total += info.st_size
                    if total > budget:
                        raise _Failure('checkpoint', 'checkpoint_byte_limit')
                    digest, size = hashlib.sha256(), 0
                    with path.open('rb') as stream:
                        while chunk := stream.read(min(1048576, budget - total + info.st_size - size + 1)):
                            size += len(chunk)
                            if size > info.st_size:
                                raise _Failure('checkpoint', 'checkpoint_changed')
                            digest.update(chunk)
                    if size != info.st_size:
                        raise _Failure('checkpoint', 'checkpoint_changed')
                    nonempty |= size > 0
                    relative = path.relative_to(directory).as_posix().encode('utf-8')
                    entries.append({'path_sha256': hashlib.sha256(relative).hexdigest(),
                                    'bytes': size, 'sha256': digest.hexdigest()})
                else:
                    raise _Failure('checkpoint', 'checkpoint_unsafe')
    walk(directory)
    if not nonempty:
        raise _Failure('checkpoint', 'checkpoint_empty')
    entries.sort(key=lambda item: item['path_sha256'])
    return {'file_count': len(entries), 'total_bytes': total,
            'sha256': hashlib.sha256(_encode(entries)).hexdigest(), 'files': entries}


def _comparison(expected, actual, phase):
    left = {'$/state' + path[1:]: leaf for path, leaf in expected['state'].items()}
    right = {'$/state' + path[1:]: leaf for path, leaf in actual['state'].items()}
    if phase != 'restore':
        for source, target in ((expected, left), (actual, right)):
            for key, value in source['metrics'].items():
                target['$/metrics/s:' + escape(key)] = fingerprint(value)['$']
    return {'phase': phase, 'step': actual['step'], **compare_fingerprints(left, right),
            'reference_metrics': expected['metrics'] if phase != 'restore' else None,
            'resumed_metrics': actual['metrics'] if phase != 'restore' else None}


def _python_launch():
    """Use the actual interpreter PID, including Windows venvs.

    CPython's Windows venv python.exe is a redirector. It starts the base
    interpreter and sets __PYVENV_LAUNCHER__ to retain the venv executable and
    prefix. Use that same documented-in-source mechanism directly, so the
    Popen handle owns the process reporting the worker PID. Do not relax PID
    validation or change the parent environment.
    See CPython v3.12.3 PC/launcher.c, VENV_REDIRECT path.
    """
    if os.name == 'nt' and sys.prefix != sys.base_prefix:
        base = getattr(sys, '_base_executable', None)
        if not base or not Path(base).is_file():
            raise OSError('base interpreter unavailable')
        return base, {**os.environ, '__PYVENV_LAUNCHER__': sys.executable}
    return sys.executable, None


def audit_resume(factory: str, *, factory_config=None, steps=8, split=4, seed=7,
                 restart_seed=991, timeout=120, max_tensor_bytes=67108864,
                 max_leaves=20000, max_trace_leaves=200000,
                 max_checkpoint_bytes=67108864) -> dict:
    """Audit finite-horizon continuation using four sequential fresh interpreters.

    Argument errors raise before any child starts; execution failures are INVALID.
    Factory configuration is represented by a digest, not echoed into the report.
    """
    config = dict(factory=factory, factory_config={} if factory_config is None else factory_config,
                  steps=steps, split=split, seed=seed, restart_seed=restart_seed,
                  timeout=timeout, max_tensor_bytes=max_tensor_bytes, max_leaves=max_leaves,
                  max_trace_leaves=max_trace_leaves, max_checkpoint_bytes=max_checkpoint_bytes)
    _validate_config(factory, config)
    config = json.loads(_encode(config))
    public_config = {k: v for k, v in config.items() if k != 'factory_config'}
    public_config['factory_config_sha256'] = hashlib.sha256(_encode(config['factory_config'])).hexdigest()
    report = {'schema_version': 1, 'status': 'INVALID', 'configuration': public_config,
              'environment': {'python': platform.python_version(), 'pytorch': version('torch'),
                              'numpy': version('numpy'), 'os': platform.system()},
              'control': {'status': 'NOT_RUN', 'steps_checked': 0}, 'workers': [],
              'first_divergence': None, 'comparisons': [], 'checkpoint': None,
              'coverage': {'requested_steps': steps, 'split': split, 'prefix_steps_checked': 0,
                           'resumed_steps_checked': 0, 'trace_leaves': 0,
                           'scope': 'deterministic CPU eager; whole optimizer steps; observed state only',
                           'limitations': ['finite horizon is not a proof of future equivalence',
                                           'observers must be pure; non-RNG side effects are not certified',
                                           'no CUDA, distributed, AMP, compiled graphs or user subprocesses']},
              'error': None}
    root = None
    inspection_history = []

    def record_inspection(value, role):
        declaration = value.get('inspection')
        if inspection_history and declaration != inspection_history[0]:
            raise _Failure(role, 'inspection_mode_changed')
        inspection_history.append(declaration)
        if declaration is not None:
            report['coverage']['inspection'] = declaration

    def run(role, start, end):
        token = uuid.uuid4().hex
        result_path, job_path = root / (role + '.result.json'), root / (role + '.job.json')
        job = {'schema_version': 1, 'token': token, 'role': role, 'configuration': config,
               'start': start, 'end': end, 'checkpoint_dir': str(root / 'checkpoint'),
               'result_path': str(result_path),
               'trace_budget': max_trace_leaves - report['coverage']['trace_leaves']}
        payload = _encode(job)
        if len(payload) > JOB_BYTES:
            raise _Failure(role, 'job_byte_limit')
        job_path.write_bytes(payload)
        worker = {'role': role, 'token': token, 'pid': None, 'exit_code': None}
        report['workers'].append(worker)
        try:
            executable, environment = _python_launch()
            child = subprocess.Popen([executable, '-m', 'resumetrace._worker', str(job_path)],
                                     env=environment, stdin=subprocess.DEVNULL,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            raise _Failure(role, 'worker_start', 'OSError') from None
        worker['pid'] = child.pid
        try:
            try:
                child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                raise _Failure(role, 'worker_timeout', 'TimeoutError') from None
        finally:
            if child.poll() is None:
                child.kill()
            worker['exit_code'] = child.wait()
        if child.returncode != 0:
            raise _Failure(role, 'worker_exit', 'RuntimeError')
        try:
            value = _read_json(result_path, RESULT_BYTES)
            observations, count = _validate_result(value, job, child.pid)
        except _Failure:
            # A worker can fail before its factory constructs an inspection wrapper.
            # Such a structured error has no declaration to compare; keep its
            # original reason instead of blaming a mode change.
            if 'inspection' in value:
                record_inspection(value, role)
            raise
        except Exception:
            raise _Failure(role, 'invalid_worker_result') from None
        record_inspection(value, role)
        report['coverage']['trace_leaves'] += count
        return observations
    try:
        root = Path(tempfile.mkdtemp(prefix='resumetrace-'))
        (root / 'checkpoint').mkdir()
        reference = run('reference', 0, steps)
        report['control']['status'] = 'INVALID'
        control = run('control', 0, steps)
        if any(not _comparison(a, b, 'restore' if i == 0 else 'step')['matched']
               for i, (a, b) in enumerate(zip(reference, control))):
            report['control']['status'] = 'INVALID'
            raise _Failure('control', 'nondeterministic_control')
        report['control'] = {'status': 'PASS', 'steps_checked': steps}
        del control
        prefix = run('prefix', 0, split)
        if any(not _comparison(reference[i], b, 'restore' if i == 0 else 'step')['matched']
               for i, b in enumerate(prefix)):
            raise _Failure('prefix', 'prefix_mismatch')
        report['coverage']['prefix_steps_checked'] = split
        del prefix
        report['checkpoint'] = _checkpoint(root / 'checkpoint', max_checkpoint_bytes)
        resumed = run('resume', split, steps)
        for index, observation in enumerate(resumed):
            comparison = _comparison(reference[observation['step']], observation,
                                     'restore' if index == 0 else 'step')
            report['comparisons'].append(comparison)
            if not comparison['matched'] and report['first_divergence'] is None:
                report['first_divergence'] = {'phase': comparison['phase'], 'step': comparison['step'],
                                             'paths': [x['path'] for x in comparison['differences']]}
        report['coverage']['resumed_steps_checked'] = steps - split
        report['status'] = 'DRIFT' if report['first_divergence'] else 'PASS'
    except _Failure as failure:
        report['error'] = failure.error
    except Exception as error:
        report['error'] = {'stage': 'engine', 'type': _safe_type(error), 'reason': 'audit_error'}
    finally:
        if root is not None:
            try:
                shutil.rmtree(root)
            except Exception as error:
                report['status'] = 'INVALID'
                report['error'] = {'stage': 'cleanup', 'type': _safe_type(error), 'reason': 'cleanup_failed'}
    if report['error'] is not None:
        report['status'] = 'INVALID'
    return report
