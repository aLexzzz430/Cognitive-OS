from __future__ import annotations

import argparse
import plistlib
from pathlib import Path
import stat
from typing import Sequence


DEFAULT_APP_NAME = "Cognitive OS"
DEFAULT_BUNDLE_ID = "ai.cognitive-os.client"


def create_macos_app_bundle(
    repo_root: str | Path,
    output_dir: str | Path,
    *,
    app_name: str = DEFAULT_APP_NAME,
    bundle_id: str = DEFAULT_BUNDLE_ID,
) -> Path:
    root = Path(repo_root).resolve()
    app_path = Path(output_dir).resolve() / f"{app_name}.app"
    contents = app_path / "Contents"
    macos = contents / "MacOS"
    resources = contents / "Resources"
    macos.mkdir(parents=True, exist_ok=True)
    resources.mkdir(parents=True, exist_ok=True)

    executable = macos / app_name
    launcher = f"""#!/bin/zsh
set -e
REPO_ROOT={str(root)!r}
cd "$REPO_ROOT"
if [ -x "$REPO_ROOT/.venv/bin/python" ]; then
  PYTHON="$REPO_ROOT/.venv/bin/python"
else
  PYTHON="$(command -v python3)"
fi
exec "$PYTHON" -m core.app.desktop_client
"""
    executable.write_text(launcher, encoding="utf-8")
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    plist = {
        "CFBundleDevelopmentRegion": "en",
        "CFBundleDisplayName": app_name,
        "CFBundleExecutable": app_name,
        "CFBundleIdentifier": bundle_id,
        "CFBundleInfoDictionaryVersion": "6.0",
        "CFBundleName": app_name,
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": "0.1.0",
        "CFBundleVersion": "1",
        "LSMinimumSystemVersion": "12.0",
        "NSHighResolutionCapable": True,
    }
    with (contents / "Info.plist").open("wb") as handle:
        plistlib.dump(plist, handle, sort_keys=True)

    readme = (
        "Cognitive OS desktop app launcher.\n\n"
        f"Source root: {root}\n"
        "This app uses .venv/bin/python when available, then falls back to python3.\n"
    )
    (resources / "README.txt").write_text(readme, encoding="utf-8")
    return app_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a lightweight macOS .app launcher for Cognitive OS.")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--output-dir", default="dist")
    parser.add_argument("--name", default=DEFAULT_APP_NAME)
    parser.add_argument("--bundle-id", default=DEFAULT_BUNDLE_ID)
    args = parser.parse_args(list(argv) if argv is not None else None)
    app_path = create_macos_app_bundle(
        args.repo_root,
        args.output_dir,
        app_name=str(args.name),
        bundle_id=str(args.bundle_id),
    )
    print(f"app_bundle={app_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
