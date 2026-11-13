"""Entry point of the installed DLNet program (PyInstaller): the same as `python -m dlnet`."""

import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()  # DataLoader workers re-enter this program on Windows / macOS
    from dlnet.cli import main

    sys.exit(main())
