#!/usr/bin/env bash
set -x
pip install causal-conv1d --no-build-isolation 2>&1 | tail -5
python3 -c "import causal_conv1d; print(\"causal_conv1d ok\", causal_conv1d.__version__)"
