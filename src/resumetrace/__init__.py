"""Cold-process checkpoint continuation auditing for deterministic CPU PyTorch."""
from pathlib import Path
from typing import Mapping, Protocol

from .engine import audit_resume

__version__ = '0.1.0'
__all__ = ['TrainingSession', 'audit_resume']


class TrainingSession(Protocol):
    """Trusted callbacks; observe must expose relevant state without side effects."""

    def step(self) -> Mapping[str, float]: ...

    def observe(self) -> Mapping: ...

    def save(self, directory: Path) -> None: ...

    def load(self, directory: Path) -> None: ...
