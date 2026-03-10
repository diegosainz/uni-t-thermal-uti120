"""Main thermal viewer application."""

from __future__ import annotations

import logging

__all__ = ["main"]


def main() -> None:
    """Entry point for the thermal viewer."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    from .gui import run_gui
    run_gui()
