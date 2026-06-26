#!/usr/bin/env bash
set -euo pipefail

export MPLBACKEND="${MPLBACKEND:-TkAgg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

SYSTEM_X11_LIBS="/usr/lib/x86_64-linux-gnu/libX11.so.6:/usr/lib/x86_64-linux-gnu/libxcb.so.1"
if [[ -n "${LD_PRELOAD:-}" ]]; then
    export LD_PRELOAD="${SYSTEM_X11_LIBS}:${LD_PRELOAD}"
else
    export LD_PRELOAD="${SYSTEM_X11_LIBS}"
fi

exec "$@"
