#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/run_supervised_indomain.sh"
bash "${SCRIPT_DIR}/run_supervised_ood.sh"
