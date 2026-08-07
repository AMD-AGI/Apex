#!/usr/bin/env bash
# Install Apex, runtime dependencies, and exact formal E2E source checkouts.

set -euo pipefail

APEX_SETUP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$APEX_SETUP_ROOT/scripts/bootstrap_dependencies.py" install "$@"
