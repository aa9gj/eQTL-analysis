"""Analysis modules for eQTL discovery."""

from eqtl_analysis.analysis.tensorqtl_runner import TensorQTLRunner
from eqtl_analysis.analysis.results import EQTLResults

__all__ = [
    "TensorQTLRunner",
    "EQTLResults",
]
