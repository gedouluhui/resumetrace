"""Cold-process checkpoint continuation auditing for deterministic CPU PyTorch."""
from pathlib import Path
from typing import Mapping, Protocol

from .engine import audit_resume
from .torch_observer import TorchSession

__version__ = '0.2.0a1'
__all__ = ['TrainingSession', 'audit_resume', 'TorchSession']


class TrainingSession(Protocol):
    """Trusted callbacks; observe must expose relevant state without side effects."""

    def step(self) -> Mapping[str, float]: ...

    def observe(self) -> Mapping: ...

    def save(self, directory: Path) -> None: ...

    def load(self, directory: Path) -> None: ...
