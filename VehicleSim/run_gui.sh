#!/usr/bin/env bash
set -euo pipefail

PYSIDE_QT="$(
python - <<'PY'
from pathlib import Path
import PySide6
print(Path(PySide6.__file__).parent / "Qt")
PY
)"

CONDA_QT_PLUGIN="$CONDA_PREFIX/lib/qt6/plugins"

# 电脑1：存在 Conda Qt/PyQt 与 PySide6 混用风险
if [[ -d "$CONDA_PREFIX/lib/qt6" ]]; then
    export LD_LIBRARY_PATH="$PYSIDE_QT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export QT_PLUGIN_PATH="$PYSIDE_QT/plugins"
    export QT_QPA_PLATFORM_PLUGIN_PATH="$PYSIDE_QT/plugins/platforms"
    export QT_QPA_PLATFORMTHEME=""
    export QT_STYLE_OVERRIDE="Fusion"
    export QT_QPA_PLATFORM="xcb"
fi

exec python main.py