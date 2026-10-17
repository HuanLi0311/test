#!/bin/bash
# Create an isolated venv for RibonanzaNet on top of the dit4dit env's torch,
# then install the packages the repo needs that the base env lacks.
set -euo pipefail

ROOT="$HOME/test/rna"
BASE_PY="$HOME/.conda/envs/dit4dit/bin/python"
VENV="$ROOT/venv"

if [ ! -d "$VENV" ]; then
    "$BASE_PY" -m venv --system-site-packages "$VENV"
fi

"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet \
    polars \
    ranger-optimizer \
    tqdm \
    scikit-learn \
    matplotlib \
    kaggle

echo "--- versions ---"
"$VENV/bin/python" -c "
import torch, polars, einops, accelerate, sklearn
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.version.cuda)
print('polars', polars.__version__)
print('einops', einops.__version__)
print('accelerate', accelerate.__version__)
print('sklearn', sklearn.__version__)
"
echo "ENV READY: $VENV/bin/python"
