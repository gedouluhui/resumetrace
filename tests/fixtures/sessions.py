"""Minimal real callbacks, fault injection, and lifecycle evidence."""
import atexit
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch


# This evidence is produced at import time, not from a harness-provided verdict.
_IMPORT_RANDOM = (random.random(), float(np.random.random()), float(torch.rand(())))
_CACHE = {}


class Session:
    def __init__(self, config):
        self.config = config
        self.fault = config.get('fault', 'none')
        self.value = 0
        self.steps_taken = 0
        self.loaded = False
        self.generator = torch.Generator().manual_seed(31)
        self.lifecycle = config.get('lifecycle')
        if self.lifecycle:
            self.event('start')
            atexit.register(self.event, 'exit')
        if self.fault == 'throw':
            raise RuntimeError('PRIVATE_EXCEPTION_TEXT C:/private/secret')
        if self.fault == 'sleep':
            time.sleep(90)
        if self.fault == 'crash':
            os._exit(17)
        if self.fault == 'control':
            self.value = os.getpid()

    def event(self, event):
        with Path(self.lifecycle).open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'event': event, 'pid': os.getpid(),
                                     'value': self.value, 'steps_taken': self.steps_taken}) + '\n')

    def step(self):
        self.steps_taken += 1
        self.value += 1
        loss = float(self.value) + float(torch.rand((), generator=self.generator))
        if self.fault == 'metric' and self.loaded:
            loss += 1
        if self.fault == 'bad_metrics':
            return {'loss': float('nan')}
        if self.fault == 'bool_metrics':
            return {'loss': True}
        return {'loss': loss}

    def observe(self):
        if self.fault == 'observer_python':
            random.random()
        if self.fault == 'observer_numpy':
            np.random.random()
        if self.fault == 'observer_torch':
            torch.rand(())
        if self.fault == 'bad_state':
            return {'bad': object()}
        if self.fault == 'cycle':
            cycle = {}; cycle['cycle'] = cycle
            return cycle
        return {'value': self.value, 'rng': self.generator.get_state(),
                'import_random': _IMPORT_RANDOM}

    def save(self, directory):
        if self.lifecycle:
            self.event('save')
        if self.fault == 'save':
            raise RuntimeError('PRIVATE_EXCEPTION_TEXT')
        if self.fault == 'empty_checkpoint':
            return
        if self.fault == 'zero_checkpoint':
            (directory / 'empty').touch()
            return
        if self.fault == 'many_checkpoint':
            for i in range(1001):
                (directory / str(i)).write_bytes(b'x')
            return
        if self.fault == 'symlink_checkpoint':
            (directory / 'link').symlink_to(directory, target_is_directory=True)
            return
        numpy_state = np.random.get_state()
        payload = {'value': self.value, 'rng': self.generator.get_state().tolist(),
                   'import_random': list(_IMPORT_RANDOM),
                   'global_rng': {'python': random.getstate(),
                                  'numpy': (numpy_state[0], numpy_state[1].tolist(),
                                            *numpy_state[2:]),
                                  'torch': torch.get_rng_state().tolist()}}
        if self.fault == 'process-cache':
            _CACHE['value'] = payload.pop('value')
        (directory / 'state.json').write_text(json.dumps(payload), encoding='utf-8')

    def load(self, directory):
        global _IMPORT_RANDOM
        self.loaded = True
        if self.lifecycle:
            self.event('load')
        if self.fault == 'load':
            raise RuntimeError('PRIVATE_EXCEPTION_TEXT')
        payload = json.loads((directory / 'state.json').read_text(encoding='utf-8'))
        if self.fault != 'drift':
            self.value = payload.get('value', _CACHE.get('value', 0))
        self.generator.set_state(torch.tensor(payload['rng'], dtype=torch.uint8))
        _IMPORT_RANDOM = tuple(payload['import_random'])
        rng = payload['global_rng']
        python_state = rng['python']
        random.setstate((python_state[0], tuple(python_state[1]), python_state[2]))
        numpy_state = rng['numpy']
        np.random.set_state((numpy_state[0], np.array(numpy_state[1], dtype=np.uint32),
                             *numpy_state[2:]))
        torch.set_rng_state(torch.tensor(rng['torch'], dtype=torch.uint8))


def make(config):
    return Session(config)
