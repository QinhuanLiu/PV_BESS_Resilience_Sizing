from __future__ import annotations

from src.app.cli import DISPATCH_BACKEND_CHOICES, main
from src.app.workflow import ORDINARY_SENSITIVITY_PARAMETERS, run_all, run_selected

__all__ = [
    "DISPATCH_BACKEND_CHOICES",
    "ORDINARY_SENSITIVITY_PARAMETERS",
    "main",
    "run_all",
    "run_selected",
]


if __name__ == "__main__":
    main()
