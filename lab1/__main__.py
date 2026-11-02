"""Точка входа для запуска пакета командой python -m lab1.

Модуль только передает управление функции main из модуля cli.
"""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
