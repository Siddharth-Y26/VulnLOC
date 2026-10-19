#!/usr/bin/env bash
# Install what the site needs, once, when the Codespace is created.
# The pins are the lock file's, with torch's CPU build and without the development tools —
# the same requirements.txt the Hugging Face image was built and tested with.
set -euo pipefail

python - > /tmp/requirements-site.txt <<'EOF'
import sys
sys.path.insert(0, "deploy/hf_space")
import stage
print(stage.requirements_from_lock(), end="")
EOF

pip install --extra-index-url https://download.pytorch.org/whl/cpu -r /tmp/requirements-site.txt
pip install --no-deps -e .
