from __future__ import annotations

import sys

from modules.local_mirror.managed_vm import main


if __name__ == "__main__":
    raise SystemExit(main(["build-helper", *sys.argv[1:]]))
